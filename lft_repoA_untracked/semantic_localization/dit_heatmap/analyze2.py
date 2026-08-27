"""Target-focus analysis v2 -- fixes three flaws in v1:
  1. v1 averaged over all 24 latent frames, but the object MOVES; overlaying that on the initial frame
     smears everything. -> use only the early latent frames aligned with the initial frame.
  2. v1's hard top-20% metric degenerated (the CLIPSeg mask covers ~4 of 720 cells). -> use a soft
     mask-weighted focus RATIO: mean saliency inside the mask / mean saliency outside.
  3. Raw feature norm is dominated by high-norm artifact tokens. -> standardize each map before use.
Reports per-block numbers, and the plan-ON minus plan-OFF difference map.
"""
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
NPZ = HERE / "dit_heatmap/dit_feats.npz"
FRAME = HERE / "oracle_repro/yc74616_f0.png"
FIGS = HERE / "figs"
RESULT = HERE / "dit_heatmap/focus_result_v2.txt"
BLOCKS = [14, 20, 27]
TEARLY = 3          # latent frames aligned with the initial frame


def clipseg(img, noun, hw):
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    pr = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    sg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval()
    inp = pr(text=[noun], images=[Image.fromarray(img).resize((352, 352))], return_tensors="pt", padding=True)
    with torch.no_grad():
        lg = sg(**inp).logits
    if lg.ndim == 2: lg = lg[None]
    m = F.interpolate(torch.sigmoid(lg[0])[None, None], hw, mode="bilinear")[0, 0].numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-6)


def z01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


def main():
    z = np.load(NPZ); keys = list(z.files)
    a0 = z[f"plan_on_b{BLOCKS[0]}"]
    H, W = int(a0.shape[2]), int(a0.shape[3])

    def grid(a):
        a = a.astype(np.float32)
        a = a[0] if a.shape[0] == 1 else a
        g = np.linalg.norm(a, axis=-1)
        if g.ndim == 1: g = g.reshape(-1, H, W)
        g = g[:TEARLY].mean(0)                       # early frames only (aligned with initial frame)
        return (g - g.mean()) / (g.std() + 1e-6)     # standardize: kill high-norm offset

    img = np.asarray(Image.open(FRAME).convert("RGB"))
    m_car, m_ban = clipseg(img, "yellow carrot", (H, W)), clipseg(img, "banana", (H, W))

    def ratio(s, m):
        """soft mask-weighted focus ratio: mean saliency inside / outside (1.0 = no preference)."""
        s = s - s.min(); mi = (s * m).sum() / (m.sum() + 1e-6); mo = (s * (1 - m)).sum() / ((1 - m).sum() + 1e-6)
        return float(mi / (mo + 1e-6))

    per_block, lines = {}, []
    for tag in ("plan_off", "plan_on", "plan_on_planattn"):
        maps = []
        for b in BLOCKS:
            k = f"{tag}_b{b}" if "planattn" not in tag else f"plan_on_planattn_b{b}"
            if k not in keys: continue
            g = grid(z[k]); maps.append(g)
            per_block[f"{tag}_b{b}"] = (ratio(g, m_car), ratio(g, m_ban))
        if maps: per_block[tag] = maps
    sal = {t: np.mean(per_block[t], axis=0) for t in ("plan_off", "plan_on", "plan_on_planattn") if t in per_block}

    lines.append("focus ratio = mean saliency inside mask / outside  (1.0 = no preference)")
    for t in sal:
        lines.append(f"{t:18s} carrot={ratio(sal[t], m_car):.3f}  banana={ratio(sal[t], m_ban):.3f}")
    lines.append("")
    lines.append("per-block:")
    for b in BLOCKS:
        for t in ("plan_off", "plan_on", "plan_on_planattn"):
            k = f"{t}_b{b}"
            if k in per_block and isinstance(per_block[k], tuple):
                lines.append(f"  block{b:2d} {t:18s} carrot={per_block[k][0]:.3f} banana={per_block[k][1]:.3f}")
    if "plan_on" in sal and "plan_off" in sal:
        d = sal["plan_on"] - sal["plan_off"]
        lines.append("")
        lines.append(f"ON-OFF diff map: carrot={ratio(d, m_car):.3f} banana={ratio(d, m_ban):.3f} "
                     f"|d|mean={np.abs(d).mean():.4f} (feat scale ~1.0 after standardize)")
    RESULT.write_text("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

    base = np.asarray(Image.fromarray(img).resize((W * 20, H * 20))).astype(float) / 255.
    def ov(s, cmap="turbo", signed=False):
        u = torch.from_numpy(s)[None, None].float()
        u = F.interpolate(u, base.shape[:2], mode="bilinear")[0, 0].numpy()
        u = (u / (np.abs(u).max() + 1e-6) / 2 + 0.5) if signed else z01(u)
        return base * 0.5 + cm.get_cmap(cmap)(u)[..., :3] * 0.5
    P = [(base, "initial frame"), (ov(m_car), "CLIPSeg target: yellow carrot")]
    for t in ("plan_off", "plan_on"):
        if t in sal: P.append((ov(sal[t]), f"{t} feats (carrot ratio {ratio(sal[t], m_car):.2f})"))
    if "plan_on" in sal and "plan_off" in sal:
        P.append((ov(sal["plan_on"] - sal["plan_off"], "bwr", True), "ON - OFF (red = plan adds focus)"))
    if "plan_on_planattn" in sal:
        P.append((ov(sal["plan_on_planattn"]), f"PLAN cross-attn (carrot {ratio(sal['plan_on_planattn'], m_car):.2f})"))
    fig, ax = plt.subplots(1, len(P), figsize=(4.4 * len(P), 3.3))
    for a, (im, t) in zip(np.atleast_1d(ax), P):
        a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
    fig.suptitle("Semantic plan and target focus, early latent frames (iter3000 SG-WAM, oracle plan)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "dit_plan_target_focus_v2.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    print("SAVED figs/dit_plan_target_focus_v2.png", flush=True); print("ANALYZE2-DONE", flush=True)


if __name__ == "__main__":
    main()
