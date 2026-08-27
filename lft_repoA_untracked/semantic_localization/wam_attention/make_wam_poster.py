"""Render the FastWAM cross-attention poster: RGB-only (diffuse) vs SG-WAM (focused).

Consolidated replacement for select_wam.py. Same scoring and colour handling, plus the two things
that were missing when the original figure had to be reproduced:
  * MATCH_PROMPTS -- pick scenes by prompt instead of by rank, so a specific figure's task line-up can
    be rebuilt even though the .npz has since been regenerated with different samples
  * VIEW=main|composite -- the main-camera set and the main|wrist set live in different .npz files

Attention maps come from generate_many.py (run once per model: `python generate_many.py rgb` and
`... sg`), which is the expensive half; this script is cheap and can be re-run freely.

  VIEW=composite TOPK=8 python make_wam_poster.py
  VIEW=main TOPK=12 FIGNAME=my_poster python make_wam_poster.py
  MATCH_PROMPTS='put the white mug on the left|put both moka pots on the stov' python make_wam_poster.py
"""
import os
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

B = os.environ.get("NPZ_DIR", "/data/LFT-W02_data/junjie/fastwam_sg_ckpt")
OUT = os.environ.get("FIG_DIR", "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/figs")
VIEW = os.environ.get("VIEW", "composite")            # composite = main|wrist, main = main camera only
# which capture run to read: "" = the frame-0 dump, "_main" = main-camera crop, "_traj" = the
# mid-trajectory frames the original poster used. VIEW only picks the default; OUT_SUFFIX wins.
SUF = os.environ.get("OUT_SUFFIX") or ("" if VIEW == "composite" else "_main")
FIGNAME = os.environ.get("FIGNAME", f"wam_sg_best{SUF}")
TOPK = int(os.environ.get("TOPK", 8))
PER_TASK = int(os.environ.get("PER_TASK", 1))         # >1 keeps several episodes of the same task
WANT = [p.strip() for p in os.environ.get("MATCH_PROMPTS", "").split("|") if p.strip()]

r = np.load(f"{B}/wam_part_rgb{SUF}.npz", allow_pickle=True)
s = np.load(f"{B}/wam_part_sg{SUF}.npz", allow_pickle=True)
N = min(len(r["maps"]), len(s["maps"]))                # the two runs share sample order
RGB, SG, FR, PR = r["maps"][:N], s["maps"][:N], r["frames"][:N], [str(x) for x in r["prompts"]][:N]
print(f"view={VIEW}  paired samples={N}  distinct tasks={len(set(PR))}", flush=True)


def conc(g):
    """top-10% mass fraction of a normalised map -- higher means more focused."""
    v = g.flatten().astype(np.float64); v = v - v.min(); p = v / (v.sum() + 1e-9)
    k = max(1, int(0.10 * len(p)))
    return float(np.sort(p)[::-1][:k].sum())


def edge_frac(g):
    """mass sitting on the 1-cell border; penalised so edge artefacts do not win the ranking."""
    gg = g - g.min(); gg = gg / (gg.sum() + 1e-9)
    b = gg.copy(); b[1:-1, 1:-1] = 0
    return float(b.sum())


def sharp(g, lo=60, hi=99, gamma=1.6):
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    plo, phi = np.percentile(g, [lo, hi])
    return np.clip((g - plo) / (phi - plo + 1e-6), 0, 1) ** gamma


TURBO = cm.get_cmap("turbo")
def ov(comp, g):
    H, W = comp.shape[:2]
    up = F.interpolate(torch.from_numpy(sharp(g))[None, None].float(), (H, W),
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    return comp.astype(float) / 255. * 0.5 + TURBO(up)[..., :3] * 0.5


scores = sorted(((conc(SG[i]) - conc(RGB[i]) - 0.5 * edge_frac(SG[i]), conc(SG[i]), conc(RGB[i]), i)
                 for i in range(N)), reverse=True)

if WANT:                                               # rebuild a specific figure's task line-up
    used, top = set(), []
    for w in WANT:
        cand = [t for t in scores if PR[t[3]].startswith(w) and t[3] not in used]
        if not cand:
            print(f"  [miss] no sample for {w!r}", flush=True); continue
        used.add(cand[0][3]); top.append(cand[0])
else:
    seen, top = {}, []
    for t in scores:
        p = PR[t[3]]
        if seen.get(p, 0) >= PER_TASK: continue
        seen[p] = seen.get(p, 0) + 1; top.append(t)
        if len(top) >= TOPK: break

for sc, cs, cr, i in top:
    print(f"  {sc:+.3f}  SG={cs:.3f} RGB={cr:.3f}  {PR[i][:48]}", flush=True)

n = len(top)
fig, ax = plt.subplots(2, n, figsize=(3.3 * n, 7))
ax = np.atleast_2d(ax)
for row, (label, M) in enumerate((("RGB-only WAM", RGB), ("SG-WAM (ours)", SG))):
    for c, (_, _, _, i) in enumerate(top):
        ax[row, c].imshow(ov(FR[i], M[i])); ax[row, c].axis("off")
        if row == 0: ax[row, c].set_title(PR[i][:30], fontsize=8)
    ax[row, 0].axis("on"); ax[row, 0].set_xticks([]); ax[row, 0].set_yticks([])
    ax[row, 0].set_ylabel(label, fontsize=13, fontweight="bold")
view_tag = "main view" if VIEW == "main" else "main|wrist"
fig.suptitle(f"FastWAM cross-attention — clearest RGB-only(diffuse) vs SG-WAM(focused) cases "
             f"(LIBERO, {view_tag})", fontsize=14)
fig.tight_layout()
path = f"{OUT}/{FIGNAME}.png"
fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)

mean_sg = float(np.mean([conc(SG[i]) for i in range(N)]))
mean_rgb = float(np.mean([conc(RGB[i]) for i in range(N)]))
print(f"SAVED {path}", flush=True)
print(f"NOTE  poster shows the {n} best-gain scenes; over ALL {N} samples the means are "
      f"SG={mean_sg:.3f} vs RGB={mean_rgb:.3f} ({mean_sg/mean_rgb:.2f}x) -- quote the means, not the poster.",
      flush=True)
print("POSTER-DONE", flush=True)
