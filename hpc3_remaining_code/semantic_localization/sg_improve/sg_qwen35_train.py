"""Real training for the Qwen3.5-2B discrete-SigLIP planner (Option 6).

Qwen3.5-2B (frozen) reads (current LIBERO frame + instruction); K*P learnable queries cross-attend to
its hidden states; a ParallelCodePlanHead predicts, per keyframe-token, a discrete code over a VQ
codebook fit to GT SigLIP2. Loss = cross-entropy vs the GT feature's nearest code. Discrete = no
mean-collapse (argmax picks a real prototype, never an average), text-grounded via SigLIP -- Plan-X's
benefit without an autoregressive generator. The 1024-d codebook vectors feed the WAM UNCHANGED.

Codebook: k-means-initialized from a GT-SigLIP sample, then FROZEN (DDP-friendly; no EMA drift across
ranks). Backbone frozen -> the clean test of "can a discrete head expose a discriminative plan".

Run: torchrun --nproc_per_node=N sg_qwen35_train.py   (WORLD_SIZE=1 also works)
"""
import os, sys, math, glob, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg_discrete_plan import SigLIPVQ, ParallelCodePlanHead, discrete_plan_loss, predict_plan_vectors
from sg_plan_losses import plan_diagnostics
from sg_qwen35_plan import PlanQueryModule
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

QWEN = os.environ.get("QWEN_DIR", "/data/user/jhe724/workspace/weights/Qwen3.5-2B")
DATA = os.environ.get("DATA_DIR", "/data/user/jhe724/workspace/VLM4WAM/data/qwen35_train")
OUT = os.environ.get("OUT_DIR", "/data/user/jhe724/workspace/VLM4WAM/outputs/qwen35_discrete_plan")
K, P, D, NUM_CODES = 4, 256, 1024, int(os.environ.get("NUM_CODES", 2048))
STEPS = int(os.environ.get("MAX_STEPS", 6000)); BS = int(os.environ.get("BATCH_SIZE", 8))
LR = float(os.environ.get("LR", 3e-4)); SAVE = int(os.environ.get("SAVE_STEPS", 1000))
FULL_FT = int(os.environ.get("FULL_FT", 0))                    # 1 = full fine-tune backbone (else frozen)
HEAD_LR = float(os.environ.get("HEAD_LR", 1e-4))              # head lr when full-FT (backbone uses LR)
WARM_KMEANS = int(os.environ.get("KMEANS_SAMPLES", 200000)); dt = torch.bfloat16


def is_main(): return int(os.environ.get("RANK", 0)) == 0
def log(m):
    if is_main(): print(m, flush=True)


def setup_ddp():
    ws = int(os.environ.get("WORLD_SIZE", 1)); rk = int(os.environ.get("RANK", 0))
    lr = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(lr)
    if ws > 1: dist.init_process_group("nccl")
    return ws, rk, lr


def load_shards():
    rgb, kf, pr = [], [], []
    for f in sorted(glob.glob(f"{DATA}/*_rgb.npy")):
        s = f[:-8]
        rgb.append(np.load(f"{s}_rgb.npy")); kf.append(np.load(f"{s}_kf.npy", mmap_mode="r"))
        pr.append(np.load(f"{s}_prompt.npy", allow_pickle=True))
    return np.concatenate(rgb), np.concatenate([np.asarray(k) for k in kf]), np.concatenate(pr)


@torch.no_grad()
def kmeans_init(vq, gt_sample, iters=15):
    """Deterministic k-means (fixed seed -> identical on every rank) on L2-normalized GT tokens."""
    x = F.normalize(gt_sample.float(), dim=-1)                       # [M,D]
    g = torch.Generator(device=x.device).manual_seed(0)
    c = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:vq.num_codes]].clone()
    for _ in range(iters):
        a = (x @ c.t()).argmax(1)                                    # assign
        for _ in range(1):
            oneh = F.one_hot(a, vq.num_codes).float()                # [M,K]
            cnt = oneh.sum(0).clamp(min=1)
            c = F.normalize((oneh.t() @ x) / cnt[:, None], dim=-1)
    vq.codebook.copy_(c)


class DiscretePlanner(nn.Module):
    def __init__(self, qwen, qdim, full_ft=False):
        super().__init__()
        self.qwen = qwen
        self.full_ft = full_ft
        if not full_ft:
            for p in qwen.parameters(): p.requires_grad_(False)   # frozen: only heads train
        else:
            qwen.gradient_checkpointing_enable()                  # full-FT: trade compute for activation memory
            if hasattr(qwen, "lm_head"):                          # lm_head unused (we read hidden states) -> no DDP-unused complaint
                for p in qwen.lm_head.parameters(): p.requires_grad_(False)
        self.query = PlanQueryModule(qdim, K * P).float()
        self.code_head = ParallelCodePlanHead(qdim, NUM_CODES).float()
        self.vq = SigLIPVQ(NUM_CODES, D)

    def forward(self, inp, gt=None):
        ctx = torch.enable_grad() if self.full_ft else torch.no_grad()
        with ctx:
            hs = self.qwen(**inp, output_hidden_states=True).hidden_states[-1].float()
        q = self.query(hs, inp.get("attention_mask"))
        logits = self.code_head(q).reshape(hs.shape[0], 1, K, P, NUM_CODES)
        if gt is None: return logits, None
        loss, _ = discrete_plan_loss(logits, gt, self.vq, ema=False)  # codebook frozen after kmeans
        return logits, loss


def make_inputs(proc, rgb_batch, prompts, dev):
    msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}] for p in prompts]
    txt = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    imgs = [Image.fromarray(r) for r in rgb_batch]
    return proc(text=txt, images=imgs, return_tensors="pt", padding=True).to(dev)


def main():
    ws, rk, lrk = setup_ddp(); dev = f"cuda:{lrk}"
    os.makedirs(OUT, exist_ok=True)
    proc = AutoProcessor.from_pretrained(QWEN)
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(QWEN, dtype=dt).to(dev)
    qwen.train() if FULL_FT else qwen.eval()
    qdim = qwen.config.text_config.hidden_size
    model = DiscretePlanner(qwen, qdim, full_ft=bool(FULL_FT)).to(dev)
    model.query = model.query.to(dev).float(); model.code_head = model.code_head.to(dev).float()

    rgb, kf, pr = load_shards()
    n = len(rgb); rng = np.random.default_rng(0); perm = rng.permutation(n)
    ntr = int(n * 0.92); tr, va = perm[:ntr], perm[ntr:]
    log(f"data N={n} train={len(tr)} val={len(va)} codes={NUM_CODES} world={ws} full_ft={FULL_FT}")

    # k-means codebook from a GT sample (same on every rank), then freeze
    m = min(WARM_KMEANS, len(tr) * K * P)
    idx = rng.choice(len(tr), min(len(tr), max(1, m // (K * P))), replace=False)
    gt_s = torch.from_numpy(np.asarray(kf[tr[idx]]).reshape(-1, D)).to(dev)
    kmeans_init(model.vq, gt_s); del gt_s; torch.cuda.empty_cache()
    log("codebook k-means initialized (frozen)")

    core = model
    if FULL_FT:
        # full fine-tune: DDP-wrap the WHOLE model (backbone + heads); split lr backbone vs head
        if ws > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[lrk], find_unused_parameters=True)
            core = model.module
        param_groups = [
            {"params": [p for p in core.qwen.parameters() if p.requires_grad], "lr": LR},
            {"params": list(core.query.parameters()) + list(core.code_head.parameters()), "lr": HEAD_LR},
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
        params = [p for g in param_groups for p in g["params"]]
        fwd = lambda inp, gt: model(inp, gt)
    else:
        # frozen backbone: only the heads train (DDP-wrap heads individually)
        if ws > 1:
            core.query = nn.parallel.DistributedDataParallel(core.query, device_ids=[lrk])
            core.code_head = nn.parallel.DistributedDataParallel(core.code_head, device_ids=[lrk])
        params = list(core.query.parameters()) + list(core.code_head.parameters())
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
        fwd = lambda inp, gt: core(inp, gt)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    grng = np.random.default_rng(1000 + rk)

    for step in range(STEPS):
        bi = tr[grng.choice(len(tr), BS, replace=False)]
        inp = make_inputs(proc, rgb[bi], [str(x) for x in pr[bi]], dev)
        gt = torch.from_numpy(np.asarray(kf[bi])).float().to(dev)[:, None]  # [B,1,K,P,D]
        _, loss = fwd(inp, gt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
        if step % 50 == 0: log(f"step {step}/{STEPS} code-CE={loss.item():.4f} lr={sch.get_last_lr()[-1]:.2e}")
        if (step + 1) % SAVE == 0 or step + 1 == STEPS:
            if is_main():
                evaluate(core, proc, rgb, kf, pr, va, dev)
                sd = {"query": core.query.state_dict(), "code_head": core.code_head.state_dict(),
                      "codebook": core.vq.codebook.cpu(), "step": step + 1,
                      "cfg": {"K": K, "P": P, "D": D, "NUM_CODES": NUM_CODES, "full_ft": FULL_FT}}
                if FULL_FT: sd["qwen"] = {k: v.cpu() for k, v in core.qwen.state_dict().items()}
                torch.save(sd, f"{OUT}/step_{step+1}.pt")
                log(f"  saved {OUT}/step_{step+1}.pt")
            if ws > 1: dist.barrier()
    log("ALLDONE")


@torch.no_grad()
def evaluate(core, proc, rgb, kf, pr, va, dev, nb=6):
    core.query.eval(); core.code_head.eval()
    preds, gts = [], []
    for i in range(0, min(len(va), nb * 8), 8):
        bi = va[i:i + 8]
        inp = make_inputs(proc, rgb[bi], [str(x) for x in pr[bi]], dev)
        logits, _ = core(inp)
        preds.append(predict_plan_vectors(logits, core.vq).cpu()); gts.append(torch.from_numpy(np.asarray(kf[bi])).float()[:, None])
    pred = torch.cat(preds); gt = torch.cat(gts)
    d = plan_diagnostics(pred.to(dev), gt.to(dev))
    log(f"  [val] retrieval@1={d['retrieval@1']:.3f} self_sim={d['self_sim']:.3f}(gt {d['gt_self_sim']:.3f}) cos_gt={d['cos_gt']:.3f}")
    core.query.train(); core.code_head.train()


if __name__ == "__main__":
    main()
