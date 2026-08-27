"""Stage 2 of the plan-loss A/B: MSE vs cosine vs InfoNCE, identical model / data / seed / steps.

Surrogate planner (a faithful miniature of the real one): K*P learnable queries cross-attend to the
CURRENT frame's frozen SigLIP2 tokens + the instruction embedding, and regress the K future keyframes'
SigLIP2 tokens. Predicting the future from the present is genuinely multi-modal, so mean-collapse
manifests here exactly as it does in the real planner -- that is the point of the harness.

Metrics on a HELD-OUT episode split:
  retrieval@1 / self_sim  -- spatial discriminability & token collapse (sg_plan_losses.plan_diagnostics)
  probe AP                -- THE metric from ../sg_probe/RESULT_honest.md: a strictly LINEAR bilinear
                             probe (token . W . text) predicting the CLIPSeg target mask. Linear on
                             purpose: a conv probe overfits and hid the gap last time.
"""
import os, sys, json, math, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg_plan_losses import plan_loss, plan_diagnostics

dev = "cuda"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ab_plan")
FIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
K, P, D, DM = 4, 256, 1024, 512
STEPS, BS, LR = 2000, 8, 3e-4
PROBE_STEPS, PROBE_LR = 600, 1e-3
ARMS = ["mse", "cosine", "infonce", "infonce+cosine"]
def log(m): print(m, flush=True)


class SurrogatePlanner(nn.Module):
    """queries -> cross-attend(current SigLIP tokens + text) -> per-keyframe-token plan [B,1,K,P,D]."""
    def __init__(self, depth=4, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(K * P, DM) * 0.02)
        self.in_proj, self.txt_proj = nn.Linear(D, DM), nn.Linear(D, DM)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "ln_q": nn.LayerNorm(DM), "ln_kv": nn.LayerNorm(DM),
            "attn": nn.MultiheadAttention(DM, heads, batch_first=True),
            "ln_f": nn.LayerNorm(DM),
            "ff": nn.Sequential(nn.Linear(DM, DM * 4), nn.GELU(), nn.Linear(DM * 4, DM)),
        }) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(DM), nn.Linear(DM, D))

    def forward(self, cur, txt):
        b = cur.shape[0]
        kv = torch.cat([self.in_proj(cur), self.txt_proj(txt)[:, None]], dim=1)   # [B, P+1, DM]
        x = self.q[None].expand(b, -1, -1)
        for l in self.layers:
            x = x + l["attn"](l["ln_q"](x), l["ln_kv"](kv), kv)[0]
            x = x + l["ff"](l["ln_f"](x))
        return self.out(x).reshape(b, 1, K, P, D)


class LinearProbe(nn.Module):
    """Strictly bilinear: logit[p] = (token_p @ W) . text. No nonlinearity -> it can only read out what
    is LINEARLY present in the plan, which is what we want to measure."""
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(D, D, bias=False)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, text):            # tokens [N,P,D], text [N,D]
        return (self.W(tokens) * F.normalize(text, dim=-1)[:, None]).sum(-1) + self.b   # [N,P]


def load_data():
    g = {k: np.load(os.path.join(DATA, f"{k}.npy")) for k in ("cur", "kf", "txt", "mask", "suite")}
    n = g["cur"].shape[0]
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    ntr = int(n * 0.75)
    t = lambda a, i: torch.from_numpy(a[i]).float()
    tr, te = perm[:ntr], perm[ntr:]
    pack = lambda i: dict(cur=t(g["cur"], i), kf=t(g["kf"], i)[:, None], txt=t(g["txt"], i),
                          mask=t(g["mask"], i))
    log(f"N={n} train={len(tr)} test={len(te)}")
    return pack(tr), pack(te)


def train_arm(kind, tr, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SurrogatePlanner().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    n = tr["cur"].shape[0]; rng = np.random.default_rng(seed)
    for s in range(STEPS):
        i = rng.choice(n, BS, replace=False)
        cur, kf, txt = (tr[k][i].to(dev) for k in ("cur", "kf", "txt"))
        loss = plan_loss(m(cur, txt), kf, kind=kind)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        if s % 500 == 0: log(f"  [{kind}] step {s}: loss={loss.item():.4f}")
    return m.eval()


@torch.no_grad()
def predict(m, split, bs=8):
    out = []
    for i in range(0, split["cur"].shape[0], bs):
        out.append(m(split["cur"][i:i + bs].to(dev), split["txt"][i:i + bs].to(dev)).cpu())
    return torch.cat(out)                        # [N,1,K,P,D]


def probe_ap(pred_tr, tr, pred_te, te, seed=0):
    """Train the linear probe on TRAIN predictions, report average precision on TEST predictions."""
    torch.manual_seed(seed)
    flat = lambda pr, sp: (pr.reshape(-1, P, D).to(dev),
                           sp["txt"][:, None].expand(-1, K, -1).reshape(-1, D).to(dev),
                           sp["mask"].reshape(-1, P).to(dev))
    xtr, ttr, ytr = flat(pred_tr, tr); xte, tte, yte = flat(pred_te, te)
    pb = LinearProbe().to(dev); opt = torch.optim.AdamW(pb.parameters(), lr=PROBE_LR, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for s in range(PROBE_STEPS):
        i = torch.from_numpy(rng.choice(xtr.shape[0], 256, replace=False)).to(dev)
        loss = F.binary_cross_entropy_with_logits(pb(xtr[i], ttr[i]), ytr[i])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sc = pb(xte, tte).reshape(-1).cpu().numpy(); gt = (yte.reshape(-1).cpu().numpy() > 0.5)
    o = np.argsort(-sc); gt = gt[o]
    tp = np.cumsum(gt); prec = tp / np.arange(1, len(gt) + 1)
    return float((prec * gt).sum() / max(gt.sum(), 1)), float(gt.mean())


def main():
    tr, te = load_data()
    rows = {}
    for kind in ARMS:
        log(f"=== arm: {kind} ===")
        m = train_arm(kind, tr)
        ptr, pte = predict(m, tr), predict(m, te)
        d = plan_diagnostics(pte.to(dev), te["kf"].to(dev))
        ap, base = probe_ap(ptr, tr, pte, te)
        rows[kind] = {**d, "probe_ap": ap}
        log(f"  -> retrieval@1={d['retrieval@1']:.3f} self_sim={d['self_sim']:.3f} "
            f"(gt {d['gt_self_sim']:.3f}) cos_gt={d['cos_gt']:.3f} probe_AP={ap:.4f}")
    # GT upper bound: probe the teacher features themselves
    ap_gt, base = probe_ap(tr["kf"], tr, te["kf"], te)
    rows["__gt_teacher__"] = {"probe_ap": ap_gt}
    rows["__random_baseline__"] = {"probe_ap": base}
    log(f"GT-teacher probe_AP={ap_gt:.4f}   random={base:.4f}")
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, "ab_plan_loss.json"), "w") as f: json.dump(rows, f, indent=2)
    log("ALLDONE")


if __name__ == "__main__":
    main()
