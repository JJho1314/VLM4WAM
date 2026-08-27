"""ReconVLA Figure-4-style comparison, built from OUR real data.

Layout (per ReconVLA Fig. 4): two rows -- Baseline on top, ours below -- each row = one attention
heatmap followed by rollout frames of the generated video.

Our comparison is a clean ablation, unlike a cross-model baseline: identical weights (iter3000 SG-WAM),
identical first frame, prompt and seed; the ONLY difference is whether the semantic plan is supplied.
  Baseline  = plan zeroed  (ablate_out/none.mp4)
  SG-WAM    = oracle plan  (ablate_out/full.mp4)
Heatmap = per-position norm of the DiT block features (blocks 14/20/27, early latent frames),
identical computation for both rows, no per-row tuning and no cherry-picking.
"""
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import av

HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
NPZ = HERE / "dit_heatmap/dit_feats.npz"
ABL = HERE / "dit_heatmap/ablate_out"
FIGS = HERE / "figs"
FRAME = HERE / "oracle_repro/yc74616_f0.png"
BLOCKS = [14, 20, 27]
TEARLY = 3
FRAME_IDX = [24, 56, 88]          # rollout frames shown after the heatmap
TASK = 'Grasp the yellow carrot in the sink and place it into the black pot (leave the banana)'


def video_frames(p, idxs):
    c = av.open(str(p)); want = set(idxs); got = {}
    for i, f in enumerate(c.decode(video=0)):
        if i in want: got[i] = np.asarray(f.to_ndarray(format="rgb24"))
        if len(got) == len(want): break
    c.close()
    return [got[i] for i in idxs if i in got]


def saliency(z, tag, H, W):
    maps = []
    for b in BLOCKS:
        k = f"{tag}_b{b}"
        if k not in z.files: continue
        a = z[k].astype(np.float32)
        a = a[0] if a.shape[0] == 1 else a
        g = np.linalg.norm(a, axis=-1)
        if g.ndim == 1: g = g.reshape(-1, H, W)
        g = g[:TEARLY].mean(0)
        maps.append((g - g.mean()) / (g.std() + 1e-6))
    return np.mean(maps, axis=0)


def main():
    z = np.load(NPZ)
    a0 = z[f"plan_on_b{BLOCKS[0]}"]
    H, W = int(a0.shape[2]), int(a0.shape[3])
    img = np.asarray(Image.open(FRAME).convert("RGB"))
    rows = [("Baseline (no plan)", "plan_off", ABL / "none.mp4"),
            ("SG-WAM (ours)", "plan_on", ABL / "full.mp4")]

    fig, ax = plt.subplots(2, 1 + len(FRAME_IDX), figsize=(3.6 * (1 + len(FRAME_IDX)), 4.6))
    for r, (label, tag, vid) in enumerate(rows):
        s = saliency(z, tag, H, W)
        u = F.interpolate(torch.from_numpy(s)[None, None].float(), img.shape[:2], mode="bilinear")[0, 0].numpy()
        u = (u - u.min()) / (u.max() - u.min() + 1e-6)
        ax[r, 0].imshow(img.astype(float) / 255. * 0.45 + cm.get_cmap("jet")(u)[..., :3] * 0.55)
        ax[r, 0].set_ylabel(label, fontsize=11, fontweight="bold")
        ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
        for c, fr in enumerate(video_frames(vid, FRAME_IDX)):
            ax[r, c + 1].imshow(fr); ax[r, c + 1].axis("off")
            if r == 0: ax[r, c + 1].set_title(f"t = {FRAME_IDX[c]}", fontsize=9)
        if r == 0: ax[r, 0].set_title("attention / feature saliency", fontsize=9)
    fig.suptitle(TASK, fontsize=11)
    fig.tight_layout()
    out = FIGS / "fig4_style_plan_on_off.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"SAVED {out}", flush=True)

    d = np.abs(saliency(z, "plan_on", H, W) - saliency(z, "plan_off", H, W)).mean()
    print(f"mean |saliency(ours) - saliency(baseline)| = {d:.4f} (maps are standardized, so ~1.0 = large)", flush=True)
    print("FIG4-DONE", flush=True)


if __name__ == "__main__":
    main()
