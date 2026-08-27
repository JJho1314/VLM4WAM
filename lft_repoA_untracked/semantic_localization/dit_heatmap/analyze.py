"""Does the semantic plan focus the world model on the TARGET object?

Evidence from the captured DiT tensors (1,T,H,W,C):
  1. plan-attention map = per-position norm of SemanticRopeCrossAttention's output (plan-ON only).
     This is the plan's direct additive contribution to each video token.
  2. plan-ON vs plan-OFF block-feature saliency (same seed/frame/steps).
  3. Control: the prompt names BOTH the target (yellow carrot) and a distractor to leave alone
     (banana). A plan that encodes the target should land on the carrot, not the banana.
Metric: target-focus = fraction of the top-20% most salient positions falling inside the CLIPSeg mask.
"""
import os
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
NPZ = HERE / "dit_heatmap/dit_feats.npz"
FRAME = HERE / "oracle_repro/yc74616_f0.png"
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
RESULT = HERE / "dit_heatmap/focus_result.txt"
BLOCKS = [14, 20, 27]
TOPFRAC = 0.20


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


def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-6)


def main():
    z = np.load(NPZ)
    keys = list(z.files)
    a0 = z[f"plan_on_b{BLOCKS[0]}"]                   # (1,T,H,W,C) -> the spatial grid
    H, W = int(a0.shape[2]), int(a0.shape[3])

    def gridify(a):
        """(1,T,H,W,C) block feats or (1,T*H*W,C) cross-attn tokens -> per-position norm (H,W)."""
        a = a.astype(np.float32)
        a = a[0] if a.shape[0] == 1 else a
        g = np.linalg.norm(a, axis=-1)                # (T,H,W) or (T*H*W,)
        if g.ndim == 1:
            g = g.reshape(-1, H, W)                   # cross-attn tokens are flattened T*H*W
        return norm01(g.mean(0))                      # mean over time

    sal = {}
    for tag in ("plan_off", "plan_on", "plan_on_planattn"):
        maps = []
        for b in BLOCKS:
            k = f"{tag}_b{b}" if "planattn" not in tag else f"plan_on_planattn_b{b}"
            if k not in keys: continue
            maps.append(gridify(z[k]))
        if maps: sal[tag] = np.mean(maps, axis=0)
    img = np.asarray(Image.open(FRAME).convert("RGB"))
    m_car = clipseg(img, "yellow carrot", (H, W))
    m_ban = clipseg(img, "banana", (H, W))

    def focus(s, m, thr=0.5):
        v = s.flatten(); mk = (m.flatten() > thr).astype(np.float32)
        k = max(1, int(TOPFRAC * len(v)))
        return float(mk[np.argsort(-v)[:k]].mean()), float(mk.mean())

    lines, F_ = [], {}
    for tag in sal:
        fc, base_c = focus(sal[tag], m_car)
        fb, base_b = focus(sal[tag], m_ban)
        F_[tag] = (fc, fb)
        lines.append(f"{tag:18s} carrot-focus={fc:.3f} (chance {base_c:.3f})  banana-focus={fb:.3f} (chance {base_b:.3f})")
    if "plan_on" in F_ and "plan_off" in F_:
        lines.append(f"carrot-focus gain (ON - OFF) = {F_['plan_on'][0] - F_['plan_off'][0]:+.3f}")
    if "plan_on_planattn" in F_:
        fc, fb = F_["plan_on_planattn"]
        lines.append(f"plan-attention target vs distractor: carrot {fc:.3f} vs banana {fb:.3f} -> ratio {fc/(fb+1e-6):.2f}x")
    RESULT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    base = np.asarray(Image.fromarray(img).resize((W * 20, H * 20))).astype(float) / 255.
    def ov(s, cmap=cm.get_cmap("turbo")):
        u = F.interpolate(torch.from_numpy(norm01(s))[None, None].float(), base.shape[:2], mode="bilinear")[0, 0].numpy()
        return base * 0.5 + cmap(u)[..., :3] * 0.5
    panels = [(base, "initial frame"),
              (ov(sal["plan_off"]), f"plan-OFF feats (carrot {F_['plan_off'][0]:.2f})"),
              (ov(sal["plan_on"]), f"plan-ON feats (carrot {F_['plan_on'][0]:.2f})")]
    if "plan_on_planattn" in sal:
        panels.append((ov(sal["plan_on_planattn"]),
                       f"PLAN cross-attn (carrot {F_['plan_on_planattn'][0]:.2f} vs banana {F_['plan_on_planattn'][1]:.2f})"))
    panels.append((ov(m_car), "CLIPSeg: yellow carrot (target)"))
    fig, ax = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4))
    for a, (im, t) in zip(ax, panels):
        a.imshow(im); a.set_title(t, fontsize=10); a.axis("off")
    fig.suptitle("Does the semantic plan focus the world model on the target? (iter3000 SG-WAM, oracle plan)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "dit_plan_target_focus.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    print("SAVED figs/dit_plan_target_focus.png", flush=True); print("ANALYZE-DONE", flush=True)


if __name__ == "__main__":
    main()
