#!/usr/bin/env python3
"""One-glance overview grid for the new-scenes eval.

Per scene row: [InstructSAM maskviz | real t~mid | real t=end | zero t=end | |real-zero| x4 t=end]
Plus per-scene real-vs-zero mean abs RGB and carrot... (generic) metrics to JSON.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import decord
import numpy as np
from PIL import Image, ImageDraw

decord.bridge.set_bridge("native")


def rd(p, n=12):
    vr = decord.VideoReader(p, num_threads=2)
    T = len(vr)
    idx = [round(i * (T - 1) / (n - 1)) for i in range(n)]
    return vr.get_batch(idx).asnumpy().astype(np.float32), idx


def lab(img, t, c):
    im = Image.fromarray(img.astype(np.uint8).copy())
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    d.text((4, 3), t[:44], fill=c)
    return np.asarray(im)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    W = args.work_dir
    out = args.out or os.path.join(W, "overview_grid.jpg")

    rows = []
    metrics = {}
    for rp in sorted(glob.glob(os.path.join(W, "videos", "*__real.mp4"))):
        ep = os.path.basename(rp)[: -len("__real.mp4")]
        zp = os.path.join(W, "videos", f"{ep}__zero.mp4")
        if not os.path.exists(zp):
            continue
        rv, idx = rd(rp)
        zv, _ = rd(zp)
        n = min(len(rv), len(zv))
        mad = float(np.abs(rv[:n] - zv[:n]).mean())
        # query label from summary
        q = ep
        summ = os.path.join(W, "summary.json")
        if os.path.exists(summ):
            for rec in json.load(open(summ)):
                if rec["episode"] == ep:
                    q = rec["query"]
                    break
        metrics[ep] = {"query": q, "real_vs_zero": round(mad, 3)}

        H, Wd = rv.shape[1:3]
        mv_path = os.path.join(W, "videos", f"{ep}__real_instructsam_mask.png")
        if os.path.exists(mv_path):
            mv = np.asarray(Image.open(mv_path).convert("RGB").resize((Wd, H)))
        else:
            mv = np.zeros((H, Wd, 3), np.uint8)
        mid, end = len(rv) // 2, len(rv) - 1
        diff = np.clip(np.abs(rv[end] - zv[end]) * 4, 0, 255)
        row = np.concatenate([
            lab(mv, f"mask | {q}", (255, 200, 0)),
            lab(rv[mid], f"real t={idx[mid]}", (120, 255, 120)),
            lab(rv[end], f"real t={idx[end]}", (120, 255, 120)),
            lab(zv[end], f"zero t={idx[end]}", (180, 180, 255)),
            lab(diff, f"|real-zero|x4 d={mad:.1f}", (255, 255, 0)),
        ], axis=1)
        rows.append(row)
        print(f"{ep}: {q!r} real_vs_zero={mad:.2f}")

    grid = np.concatenate(rows, axis=0).astype(np.uint8)
    im = Image.fromarray(grid)
    im.thumbnail((2000, 4600))
    im.save(out, quality=86)
    json.dump(metrics, open(os.path.join(W, "overview_metrics.json"), "w"), indent=2)
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
