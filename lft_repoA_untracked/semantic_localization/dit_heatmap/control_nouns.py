"""Noise floor for the sharpness focus ratio.

The corrected run gives carrot a tiny edge over banana (+0.03 / +0.01). Is that meaningful or noise?
Measure the same ratio for several control nouns -- objects the plan has no reason to single out --
and for random masks matched in area. If carrot sits inside that spread, the edge is noise.
"""
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

HERE = Path("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization")
NPZ = HERE / "dit_heatmap/plan_attn.npz"
FRAME = HERE / "oracle_repro/yc74616_f0.png"
OUT = HERE / "dit_heatmap/control_result.txt"
H, W, TEARLY = 20, 36, 3
NOUNS = ["yellow carrot", "banana", "sink", "black pot", "robot arm", "faucet", "countertop", "stove"]


def main():
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    pr = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    sg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval()
    img = np.asarray(Image.open(FRAME).convert("RGB"))
    pil = Image.fromarray(img).resize((352, 352))

    def mask(noun):
        inp = pr(text=[noun], images=[pil], return_tensors="pt", padding=True)
        with torch.no_grad():
            lg = sg(**inp).logits
        if lg.ndim == 2: lg = lg[None]
        m = F.interpolate(torch.sigmoid(lg[0])[None, None], (H, W), mode="bilinear")[0, 0].numpy()
        return (m - m.min()) / (m.max() - m.min() + 1e-6)

    def ratio(s, m):
        s = s - s.min()
        mi = (s * m).sum() / (m.sum() + 1e-6); mo = (s * (1 - m)).sum() / ((1 - m).sum() + 1e-6)
        return float(mi / (mo + 1e-6))

    z = np.load(NPZ)
    blocks = sorted({int(k.split("_b")[1]) for k in z.files if k.startswith("ent_b")})
    lines = ["sharpness focus ratio per noun (1.0 = no preference); the target is 'yellow carrot'"]
    for b in blocks:
        ent = z[f"ent_b{b}"][-1]
        Sq = ent.size; T = Sq // (H * W)
        sharp = -ent.reshape(T, H, W)[:TEARLY].mean(0)
        vals = {n: ratio(sharp, mask(n)) for n in NOUNS}
        rng = np.random.default_rng(0)
        rnd = [ratio(sharp, (rng.random((H, W)) < 0.15).astype(np.float32)) for _ in range(30)]
        lines.append(f"\nblock{b}:")
        for n, v in sorted(vals.items(), key=lambda kv: -kv[1]):
            tag = "  <-- TARGET" if n == "yellow carrot" else ""
            lines.append(f"  {n:14s} {v:.3f}{tag}")
        ctrl = [v for n, v in vals.items() if n != "yellow carrot"]
        lines.append(f"  controls: mean={np.mean(ctrl):.3f} sd={np.std(ctrl):.3f} "
                     f"range=[{min(ctrl):.3f},{max(ctrl):.3f}]")
        lines.append(f"  random masks: mean={np.mean(rnd):.3f} sd={np.std(rnd):.3f}")
        zc = (vals["yellow carrot"] - np.mean(ctrl)) / (np.std(ctrl) + 1e-9)
        lines.append(f"  target z-score vs controls = {zc:+.2f}")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("CONTROL-DONE", flush=True)


if __name__ == "__main__":
    main()
