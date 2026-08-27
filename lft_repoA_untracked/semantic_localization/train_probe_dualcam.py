"""Train a loc-head ON THIS CHECKPOINT's predicted plan, and score it against CLIPSeg ground truth.

The first pass reused a loc-head trained for the older K=1 single-camera run, so its numbers were
pessimistic and could only say "how close to the teacher's argmax". This trains a head on the actual
predicted features and scores with an ABSOLUTE metric -- the CLIPSeg mask value at the head's peak
(peak-hit) plus average precision -- so the question becomes "can the target be localised from the
predicted plan", not "does it look like the teacher".

Three arms share the head architecture, the split and the schedule; only the input features differ:
  teacher   : SigLIP2 of the real keyframe        -> upper bound (the information is definitely there)
  predicted : the planner's predicted plan         -> what we actually care about
  shuffled  : predicted features from a DIFFERENT sample -> chance level for this metric
Split is BY SAMPLE, so no keyframe of a training episode appears in the test set.
"""
import os, json, sys
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_noun import target_noun
dev = "cuda"
HERE = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization"
NPZ = os.environ.get("NPZ", f"{HERE}/data/planner_feats_dualcam_k4_big.npz")
OUT = f"{HERE}/figs/dualcam_probe"; os.makedirs(OUT, exist_ok=True)
SIG = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-large-patch16-256"
GRID, DIM, RES = 16, 1024, 256
STEPS = int(os.environ.get("STEPS", 1500)); BS = int(os.environ.get("BS", 32))
CAM_IDX = [int(x) for x in os.environ.get("CAM_IDX", "0,1").split(",")]   # 0=main, 1=wrist
from transformers import AutoModel
from transformers.models.clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor


class LocHead(nn.Module):
    def __init__(self, d=DIM, hid=256, tdim=512):
        super().__init__()
        self.film = nn.Linear(tdim, 2 * hid); self.inp = nn.Conv2d(d, hid, 1)
        self.net = nn.Sequential(nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.GELU(),
                                 nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.GELU())
        self.out = nn.Conv2d(hid, 1, 1)

    def forward(self, f, t):
        x = self.inp(f.permute(0, 2, 1).reshape(f.shape[0], -1, GRID, GRID))
        g, b = self.film(t).chunk(2, -1)
        x = x * (1 + g[..., None, None]) + b[..., None, None]
        return self.out(self.net(x)).squeeze(1)


def main():
    sig = AutoModel.from_pretrained(SIG).eval().to(dev)
    cproc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    cseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
    for m in (sig, cseg):
        for p in m.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def temb(w):
        i = cproc(text=[f"a photo of a {w}"], return_tensors="pt", padding="max_length", max_length=77)
        i = {k: v.to(dev) for k, v in i.items() if k in ("input_ids", "attention_mask")}
        o = cseg.clip.text_model(**i)
        t = o.pooler_output if getattr(o, "pooler_output", None) is not None else o[1]
        return F.normalize(t.float(), dim=-1)[0]

    @torch.no_grad()
    def mask_of(img, noun):
        x = Image.fromarray(img).resize((352, 352))
        i = cproc(text=[noun], images=[x], return_tensors="pt", padding=True)
        i = {k: v.to(dev) for k, v in i.items()}
        lg = cseg(**i).logits
        if lg.ndim == 2: lg = lg[None]
        m = F.interpolate(torch.sigmoid(lg[0])[None, None], (GRID, GRID), mode="bilinear")[0, 0]
        return ((m - m.min()) / (m.max() - m.min() + 1e-6)).cpu()

    @torch.no_grad()
    def teacher(img):
        x = torch.from_numpy(np.ascontiguousarray(img)).to(dev).permute(2, 0, 1)[None].float() / 255.
        x = (F.interpolate(x, (RES, RES), mode="bilinear", align_corners=False) - 0.5) / 0.5
        o = sig.vision_model(pixel_values=x).last_hidden_state
        return (o[:, 1:] if o.shape[1] == GRID * GRID + 1 else o)[0].float().cpu()

    z = np.load(NPZ, allow_pickle=True)
    FUT, FP = z["fut"], z["fp"]
    # recompute from the prompt: the stored nouns used the old first-hit rule, which
    # returned the destination or a modifier ("basket" for "orange juice", "yellow" for mug)
    NOUNS = [target_noun(str(p), str(n).split("|")[0]) for p, n in zip(z["prompts"], z["nouns"])]
    meta = json.loads(str(z["meta"])); K = int(meta["num_keyframes"]); V = int(meta["num_camera_views"])
    N = len(FUT)
    print(f"samples={N} K={K} V={V} using cameras={CAM_IDX}", flush=True)

    feats_t, feats_p, masks, tembs, owner = [], [], [], [], []
    cache = {}
    for i in range(N):
        noun = NOUNS[i]
        if noun not in cache: cache[noun] = temb(noun)
        for v in CAM_IDX:
            for k in range(K):
                fr = FUT[i][k][v]
                feats_t.append(teacher(fr))
                feats_p.append(torch.from_numpy(FP[i][v].astype(np.float32))[k * GRID * GRID:(k + 1) * GRID * GRID])
                masks.append(mask_of(fr, noun)); tembs.append(cache[noun]); owner.append(i)
        if (i + 1) % 20 == 0: print(f"  featurised {i+1}/{N}", flush=True)
    Ft = torch.stack(feats_t); Fp = torch.stack(feats_p)
    M = torch.stack(masks); Tv = torch.stack(tembs).cpu(); own = np.array(owner)
    print(f"examples={len(M)}  feat{tuple(Fp.shape)}  mask{tuple(M.shape)}", flush=True)

    rng = np.random.default_rng(0); perm = rng.permutation(N)
    te_s = set(perm[: max(1, int(N * 0.25))].tolist())          # split BY SAMPLE, not by frame
    te = np.array([j for j in range(len(own)) if own[j] in te_s])
    tr = np.array([j for j in range(len(own)) if own[j] not in te_s])
    print(f"train={len(tr)} test={len(te)} (held-out samples={len(te_s)}/{N})", flush=True)

    def run(F_all, tag, shuffle=False, save_as=None):
        torch.manual_seed(0)
        src = F_all.clone()
        if shuffle:                                            # break the feature/frame correspondence
            g = np.random.default_rng(1); src = src[g.permutation(len(src))]
        head = LocHead().to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        r = np.random.default_rng(0)
        for s in range(STEPS):
            b = tr[r.choice(len(tr), BS, replace=False)]
            lo = head(src[b].to(dev), Tv[b].to(dev))
            loss = F.binary_cross_entropy_with_logits(lo, M[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval(); hits, aps = [], []
        with torch.no_grad():
            for i0 in range(0, len(te), 64):
                b = te[i0:i0 + 64]
                lo = head(src[b].to(dev), Tv[b].to(dev)).cpu()
                for j, bi in enumerate(b):
                    h = lo[j].numpy(); m = M[bi].numpy()
                    hits.append(float(m.flatten()[h.argmax()]))          # peak-hit: GT mask at the peak
                    o = np.argsort(-h.flatten()); g = (m.flatten()[o] > 0.5)
                    if g.sum():
                        tp = np.cumsum(g); prec = tp / np.arange(1, len(g) + 1)
                        aps.append(float((prec * g).sum() / g.sum()))
        if save_as:
            torch.save({"state": head.state_dict(), "tag": tag}, f"{OUT}/{save_as}")
        return dict(tag=tag, peak_hit=float(np.mean(hits)), ap=float(np.mean(aps) if aps else 0.0))

    rows = [run(Ft, "teacher SigLIP2", save_as="head_teacher.pt"),
        run(Fp, "PREDICTED plan", save_as="head_predicted.pt"),
        run(Fp, "shuffled (chance)", shuffle=True)]
    json.dump({"test_samples": sorted(te_s)}, open(f"{OUT}/split.json", "w"))
    base = float((M.numpy() > 0.5).mean())
    lines = [f"loc-head trained on THIS checkpoint's features; split by sample; test examples={len(te)}",
             f"mask positive rate (chance for AP) = {base:.4f}",
             f"{'arm':>22} {'peak-hit':>9} {'AP':>7}"]
    for r in rows: lines.append(f"{r['tag']:>22} {r['peak_hit']:>9.4f} {r['ap']:>7.4f}")
    txt = "\n".join(lines) + "\n"
    open(f"{OUT}/probe_trained_result.txt", "w").write(txt)
    print(txt, flush=True); print("PROBE-TRAIN-DONE", flush=True)


if __name__ == "__main__":
    main()
