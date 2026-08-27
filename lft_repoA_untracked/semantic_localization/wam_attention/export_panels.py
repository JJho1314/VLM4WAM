"""Export every panel of wam_sg_best_main.png as its own image file.

Re-renders from the saved attention maps instead of cropping the poster, so each panel comes out at
full resolution with no titles, axes or padding -- ready to drop into a paper or slide. Selection and
colour handling repeat select_wam.py exactly, so the panels match the poster one for one.

Per selected scene: frame.png (clean input), rgbonly.png, sg.png (overlays), plus a side-by-side.
"""
import os, re
import numpy as np, torch, torch.nn.functional as F
from matplotlib import cm
from PIL import Image

B = "/data/LFT-W02_data/junjie/fastwam_sg_ckpt"
SUF = os.environ.get("OUT_SUFFIX", "_main")
OUT = os.environ.get("PANEL_DIR",
                     "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/figs/wam_sg_best_main_panels")
TOPK = int(os.environ.get("TOPK", 12))
UPSCALE = int(os.environ.get("UPSCALE", 3))          # panels are 224x224; upscale for print

r = np.load(f"{B}/wam_part_rgb{SUF}.npz", allow_pickle=True)
s = np.load(f"{B}/wam_part_sg{SUF}.npz", allow_pickle=True)
N = min(len(r["maps"]), len(s["maps"]))
RGB, SG, FR, PR = r["maps"][:N], s["maps"][:N], r["frames"][:N], [str(x) for x in r["prompts"]][:N]

# The composite set stores main|wrist side by side (224x448 frames, 14x28 maps). CROP_MAIN keeps the
# left half, which is the main view of exactly the episodes in the composite poster -- the main-only
# .npz holds different episodes, so cropping is the way to get these particular scenes.
if int(os.environ.get("CROP_MAIN", 0)):
    w_f, w_m = FR.shape[2] // 2, RGB.shape[2] // 2
    FR, RGB, SG = FR[:, :, :w_f], RGB[:, :, :w_m], SG[:, :, :w_m]
    print(f"CROP_MAIN: frames -> {FR.shape[1:]}, maps -> {RGB.shape[1:]}", flush=True)


def conc(g):
    v = g.flatten().astype(np.float64); v = v - v.min(); p = v / (v.sum() + 1e-9)
    k = max(1, int(0.10 * len(p))); return float(np.sort(p)[::-1][:k].sum())


def edge_frac(g):
    gg = (g - g.min()); gg = gg / (gg.sum() + 1e-9); b = gg.copy(); b[1:-1, 1:-1] = 0; return float(b.sum())


def sharp(g, lo=60, hi=99, gamma=1.6):
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)
    plo, phi = np.percentile(g, [lo, hi])
    return (np.clip((g - plo) / (phi - plo + 1e-6), 0, 1)) ** gamma


TURBO = cm.get_cmap("turbo")


def overlay(comp, g):
    H, W = comp.shape[:2]
    up = F.interpolate(torch.from_numpy(sharp(g))[None, None].float(), (H, W),
                       mode="bilinear", align_corners=False)[0, 0].numpy()
    return comp.astype(float) / 255. * 0.5 + TURBO(up)[..., :3] * 0.5


def save(arr, path):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.dtype != np.uint8 else arr)
    if UPSCALE > 1:
        im = im.resize((im.width * UPSCALE, im.height * UPSCALE), Image.LANCZOS)
    im.save(path)


def slug(p):
    return re.sub(r"[^a-z0-9]+", "-", p.lower()).strip("-")[:48]


def main():
    scores = []
    for i in range(N):
        cs, cr = conc(SG[i]), conc(RGB[i])
        scores.append((cs - cr - 0.5 * edge_frac(SG[i]), cs, cr, i))
    scores.sort(reverse=True)

    want = [p for p in os.environ.get("MATCH_PROMPTS", "").split("|") if p.strip()]
    if want:
        # Reproduce a specific figure's task line-up: for each requested prompt prefix take its
        # best-scoring sample (a repeated prefix takes the next best, matching duplicated columns).
        used, top = set(), []
        for w in want:
            cand = [(sc, cs, cr, i) for sc, cs, cr, i in scores
                    if PR[i].startswith(w.strip()) and i not in used]
            if not cand:
                print(f"  [miss] no sample for: {w.strip()!r}", flush=True); continue
            sc, cs, cr, i = cand[0]; used.add(i); top.append((i, sc, cs, cr))
    else:
        # ranked sweep; PER_TASK>1 keeps several episodes of the same task (different object layouts)
        per_task = int(os.environ.get("PER_TASK", 1))
        seen, top = {}, []
        for sc, cs, cr, i in scores:
            if seen.get(PR[i], 0) >= per_task: continue
            seen[PR[i]] = seen.get(PR[i], 0) + 1
            top.append((i, sc, cs, cr))
            if len(top) >= TOPK: break

    os.makedirs(OUT, exist_ok=True)
    lines = ["idx\tpanel\tSG_conc\tRGB_conc\tgain\tprompt"]
    for rank, (i, sc, cs, cr) in enumerate(top, 1):
        name = f"{rank:03d}_s{i:03d}_{slug(PR[i])}"   # sample id keeps repeats of one task distinct
        d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
        save(FR[i], f"{d}/frame.png")
        save(overlay(FR[i], RGB[i]), f"{d}/rgbonly.png")
        save(overlay(FR[i], SG[i]), f"{d}/sg.png")
        if int(os.environ.get("PAIR", 0)):        # off by default: keep panels strictly single-image
            pair = np.concatenate([overlay(FR[i], RGB[i]), overlay(FR[i], SG[i])], axis=1)
            save(pair, f"{d}/pair_rgbonly_vs_sg.png")
        lines.append(f"{i}\t{name}\t{cs:.4f}\t{cr:.4f}\t{sc:+.4f}\t{PR[i]}")
        print(f"[{rank:02d}] SG={cs:.3f} RGB={cr:.3f} {PR[i][:52]}", flush=True)
    with open(os.path.join(OUT, "index.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"SAVED {len(top)} panels -> {OUT}", flush=True)
    print("EXPORT-DONE", flush=True)


if __name__ == "__main__":
    main()
