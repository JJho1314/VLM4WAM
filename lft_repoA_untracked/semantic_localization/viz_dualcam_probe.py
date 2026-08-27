"""LOCAL: probe the dual-camera K=4 planner's PREDICTED SigLIP2 plan for target location.

Reuses the already-trained loc-head (FiLM(text) + conv decoder on 1024-d SigLIP2 tokens, 16x16 grid),
which matches this checkpoint's plan spec exactly, so no retraining is needed. For each sample and
each camera the predicted plan is (K*256, 1024); it is split back into the K keyframes and each is
localised against the instruction's target noun.

Per figure row: the real keyframe, the loc-head readout on the TEACHER SigLIP2 features of that frame
(upper bound), and the readout on the planner's PREDICTED plan for the same keyframe. If the predicted
column lights up on the object, the planner's output carries usable object location.
"""
import os, sys, json
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_noun import target_noun
dev = "cuda"
HERE = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization"
NPZ = os.environ.get("NPZ", f"{HERE}/data/planner_feats_dualcam_k4_big.npz")
FIGS = f"{HERE}/figs/dualcam_probe"; os.makedirs(FIGS, exist_ok=True)
SIG = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-large-patch16-256"
GRID, DIM, RES = 16, 1024, 256
CAM_NAMES = ("main", "wrist")
CAM_IDX = [int(x) for x in os.environ.get("CAM_IDX", "0").split(",")]   # 0=main, 1=wrist
MAXN = int(os.environ.get("MAXN", 8))
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
        B = f.shape[0]
        x = self.inp(f.permute(0, 2, 1).reshape(B, -1, GRID, GRID))
        g, b = self.film(t).chunk(2, -1)
        x = x * (1 + g[..., None, None]) + b[..., None, None]
        return self.out(self.net(x)).squeeze(1)


def main():
    sig = AutoModel.from_pretrained(SIG).eval().to(dev)
    cproc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    cseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
    for m in (sig, cseg):
        for p in m.parameters(): p.requires_grad_(False)
    def load_head(name, fallback):
        p = f"{FIGS}/{name}"
        sd = torch.load(p if os.path.exists(p) else fallback, map_location=dev)["state"]
        h = LocHead().to(dev); h.load_state_dict(sd); h.eval(); return h, os.path.exists(p)
    head_t, ok_t = load_head("head_teacher.pt", f"{HERE}/loc_head.pt")
    head_p, ok_p = load_head("head_predicted.pt", f"{HERE}/loc_head.pt")
    print(f"heads loaded (trained-on-this-ckpt: teacher={ok_t} predicted={ok_p})", flush=True)
    # only visualise HELD-OUT samples: these heads never saw them during training
    split = f"{FIGS}/split.json"
    HELD = set(json.load(open(split))["test_samples"]) if os.path.exists(split) else None

    @torch.no_grad()
    def temb(w):
        inp = cproc(text=[f"a photo of a {w}"], return_tensors="pt", padding="max_length", max_length=77)
        inp = {k: v.to(dev) for k, v in inp.items() if k in ("input_ids", "attention_mask")}
        o = cseg.clip.text_model(**inp)
        t = o.pooler_output if getattr(o, "pooler_output", None) is not None else o[1]
        return F.normalize(t.float(), dim=-1)

    @torch.no_grad()
    def teacher(img):
        x = torch.from_numpy(np.ascontiguousarray(img)).to(dev).permute(2, 0, 1)[None].float() / 255.
        x = (F.interpolate(x, (RES, RES), mode="bilinear", align_corners=False) - 0.5) / 0.5
        o = sig.vision_model(pixel_values=x).last_hidden_state
        return o[:, 1:] if o.shape[1] == GRID * GRID + 1 else o

    z = np.load(NPZ, allow_pickle=True)
    CUR, FUT, FP = z["cur"], z["fut"], z["fp"]
    PROMPTS = [str(x) for x in z["prompts"]]
    NOUNS = [[target_noun(str(p), str(n).split("|")[0])] for p, n in zip(z["prompts"], z["nouns"])]
    SUITES = [str(x) for x in z["suites"]]
    meta = json.loads(str(z["meta"])); K = int(meta["num_keyframes"])
    print(f"samples={len(CUR)} K={K} cams={meta['num_camera_views']} offsets={meta['offsets']}", flush=True)

    TURBO = cm.get_cmap("turbo")
    def ov(img, hm, lo=55.0, hi=99.5, gamma=1.9):
        h = torch.from_numpy(hm)[None, None].float()
        h = F.interpolate(h, img.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
        plo, phi = np.percentile(h, [lo, hi])
        h = np.clip((h - plo) / (phi - plo + 1e-6), 0, 1) ** gamma
        return img.astype(float) / 255. * 0.5 + TURBO(h)[..., :3] * 0.5

    stats = []
    order = [i for i in range(len(CUR)) if HELD is None or i in HELD]
    print(f"visualising {min(len(order), MAXN)} held-out samples", flush=True)
    for i in order[:MAXN]:
        noun = NOUNS[i][0]
        t = temb(noun)
        for v in CAM_IDX:
            cam = CAM_NAMES[v]
            cols = []
            for k in range(K):
                frame = FUT[i][k][v]
                tf = teacher(frame).float()
                pf = torch.from_numpy(FP[i][v].astype(np.float32))[k * GRID * GRID:(k + 1) * GRID * GRID][None].to(dev)
                with torch.no_grad():
                    h_t = head_t(tf, t)[0].cpu().numpy()
                    h_p = head_p(pf, t)[0].cpu().numpy()
                cols.append((frame, h_t, h_p))
                # agreement between teacher and predicted readout (argmax distance in grid cells)
                at = np.unravel_index(h_t.argmax(), h_t.shape); ap = np.unravel_index(h_p.argmax(), h_p.shape)
                stats.append(dict(sample=i, cam=cam, k=k, noun=noun,
                                  dist=float(np.hypot(at[0] - ap[0], at[1] - ap[1]))))
            fig, ax = plt.subplots(3, K, figsize=(3.1 * K, 9.2))
            for k, (frame, h_t, h_p) in enumerate(cols):
                ax[0, k].imshow(frame); ax[0, k].set_title(f"keyframe {k} (t+{meta['offsets'][k]})", fontsize=9)
                ax[1, k].imshow(ov(frame, h_t))
                ax[2, k].imshow(ov(frame, h_p))
                for r in range(3): ax[r, k].axis("off")
            for r, lab in enumerate(("real keyframe", f"teacher SigLIP2 -> loc-head", "PREDICTED plan -> loc-head")):
                ax[r, 0].axis("on"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
                ax[r, 0].set_ylabel(lab, fontsize=10, fontweight="bold")
            fig.suptitle(f"[{SUITES[i]}] {PROMPTS[i][:80]}   |  target='{noun}'  camera={cam}", fontsize=11)
            fig.tight_layout()
            out = f"{FIGS}/{i:02d}_{SUITES[i]}_{cam}_{noun}.png"
            fig.savefig(out, dpi=105, bbox_inches="tight"); plt.close(fig)
        print(f"[sample {i}] {SUITES[i]} '{noun}' saved (held-out)", flush=True)

    d = np.array([s["dist"] for s in stats])
    summary = (f"teacher-vs-predicted argmax distance on the {GRID}x{GRID} grid over {len(d)} "
               f"(sample,camera,keyframe) triples:\n  mean={d.mean():.2f} median={np.median(d):.2f} "
               f"cells;  within 2 cells: {(d <= 2).mean()*100:.1f}%  within 4: {(d <= 4).mean()*100:.1f}%\n"
               f"  (random pairs of points on a 16x16 grid average ~8.4 cells apart)\n")
    open(f"{FIGS}/summary.txt", "w").write(summary)
    print(summary, flush=True); print(f"SAVED -> {FIGS}", flush=True); print("VIZ-DONE", flush=True)


if __name__ == "__main__":
    main()
