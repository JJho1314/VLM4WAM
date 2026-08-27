"""Per-block target focus for the SG-WAM, scored exactly like the mask-guided reference pack.

Reference numbers to compare against (2b_mgv3_target_context, mask-supervised, sample_000):
    blocks 12 / 16 -> inside/outside ratio 37.7 / 46.0, normalised entropy 0.705 / 0.676
    all other blocks -> ratio ~1.0, entropy ~1.0 (uniform)
That model focuses ONLY in the middle blocks, which is why sweeping every block matters here.

Emits the reference's four-panel figure (initial frame | target mask | cross-attention | overlay)
for the best block of each attention path, plus the full per-block table.
"""
import os
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
NPZ = HERE / "dit_heatmap/allblocks/attn_on.npz"
FRAME = HERE / "oracle_repro/yc74616_f0.png"
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
RESULT = HERE / "dit_heatmap/allblocks/per_block.txt"
TARGET = os.environ.get("TARGET_NOUN", "yellow carrot")
TEARLY = int(os.environ.get("TEARLY", 3))     # latent frames aligned with the initial frame


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


def metrics(att, mask_bin):
    """att (H,W) attention, mask_bin (H,W) 0/1 -> reference-style metrics."""
    a = att.astype(np.float64); a = a - a.min()
    p = a / (a.sum() + 1e-12)
    inside = mask_bin > 0.5
    mass = float(p[inside].sum())
    mi = float(a[inside].mean()) if inside.any() else 0.0
    mo = float(a[~inside].mean()) if (~inside).any() else 0.0
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))     # normalised entropy
    py, px = np.unravel_index(a.argmax(), a.shape)
    return dict(mass_inside=mass, ratio=mi / (mo + 1e-12), entropy=ent,
                peak=(int(py), int(px)), peak_inside=float(inside[py, px]))


def main():
    z = np.load(NPZ)
    T, H, W = [int(v) for v in z["thw"]]
    img = np.asarray(Image.open(FRAME).convert("RGB"))
    soft = clipseg(img, TARGET, (H, W))
    thr = float(np.quantile(soft, 0.97))                # keep the mask small, like the reference (~1.7% area)
    mask = (soft > thr).astype(np.float32)
    lines = [f"grid T={T} H={H} W={W}   target='{TARGET}'   mask area={mask.mean():.4f}",
             "reference (mask-supervised model): blocks 12/16 ratio 37.7/46.0 entropy 0.705/0.676, others ~1.0",
             f"{'kind':>5} {'block':>5} {'ratio':>9} {'entropy':>8} {'mass_in':>8} {'peak_in':>7}"]
    best = {}
    for kind in ("text", "plan"):
        for b in range(28):
            k = f"{kind}_b{b}"
            if k not in z.files: continue
            att = z[k][:TEARLY].mean(0)                 # early latent frames -> aligned with initial frame
            m = metrics(att, mask)
            lines.append(f"{kind:>5} {b:>5} {m['ratio']:>9.3f} {m['entropy']:>8.4f} "
                         f"{m['mass_inside']:>8.4f} {m['peak_inside']:>7.1f}")
            if m["ratio"] > best.get(kind, (0, None, None))[0]:
                best[kind] = (m["ratio"], b, att)
    lines.append("")
    for kind, (r, b, _) in best.items():
        lines.append(f"BEST {kind}: block {b}, inside/outside ratio {r:.3f}")
    RESULT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # reference-style four-panel figure per attention path
    base = np.asarray(Image.fromarray(img).resize((W * 24, H * 24))).astype(float) / 255.
    def up(a):
        u = F.interpolate(torch.from_numpy(a.astype(np.float32))[None, None], base.shape[:2],
                          mode="bilinear")[0, 0].numpy()
        return (u - u.min()) / (u.max() - u.min() + 1e-6)
    for kind, (r, b, att) in best.items():
        hm = up(att); mk = up(mask)
        panels = [(base, "initial frame"),
                  (base * 0.5 + np.stack([mk, mk * 0, mk * 0], -1) * 0.5, f"target mask ({TARGET})"),
                  (cm.get_cmap("jet")(hm)[..., :3], f"{kind} cross-attention (block {b})"),
                  (base * 0.5 + cm.get_cmap("jet")(hm)[..., :3] * 0.5, f"overlay  ratio={r:.2f}")]
        fig, ax = plt.subplots(1, 4, figsize=(4 * 5.0, 3.0))
        for a_, (im, t) in zip(ax, panels):
            a_.imshow(np.clip(im, 0, 1)); a_.set_title(t, fontsize=10); a_.axis("off")
        fig.tight_layout()
        out = FIGS / f"allblocks_{kind}_best_block{b}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"SAVED {out}", flush=True)
    print("ANALYZE-ALLBLOCKS-DONE", flush=True)


if __name__ == "__main__":
    main()
