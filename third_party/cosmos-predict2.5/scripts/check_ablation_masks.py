#!/usr/bin/env python3
"""Visualize per-scene InstructSAM masks for the target-swap ablation.

For each scene: first frame with the REAL mask (green) and DISTRACTOR mask
(red) overlaid + bounding boxes per connected blob. One row per scene, saved as
a grid JPG. Also prints per-mask blob stats so misgrounded masks stand out.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def overlay(base: np.ndarray, mask_path: str, color, tag: str):
    im = Image.fromarray(base.copy())
    d = ImageDraw.Draw(im)
    stats = []
    if os.path.exists(mask_path):
        m = np.asarray(Image.open(mask_path).convert("L"))
        if m.shape != base.shape[:2]:
            m = np.asarray(Image.fromarray(m).resize((base.shape[1], base.shape[0]), Image.NEAREST))
        mb = m > 127
        arr = np.array(im)  # writable copy
        arr[mb] = (0.35 * arr[mb] + 0.65 * np.array(color)).astype(np.uint8)
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        lbl, n = ndimage.label(mb)
        for i in range(1, n + 1):
            ys, xs = np.where(lbl == i)
            if len(ys) < 30:
                continue
            d.rectangle([int(xs.min()) - 4, int(ys.min()) - 4, int(xs.max()) + 4, int(ys.max()) + 4],
                        outline=tuple(color), width=4)
            stats.append({"px": int(len(ys)),
                          "bbox": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]})
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d.text((5, 4), tag[:70], fill=tuple(color))
    return np.asarray(im), stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True, help="Ablation work dir (inputs/, features/, summary.json).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    work = args.work_dir
    summary = json.load(open(os.path.join(work, "summary.json")))

    rows = []
    report = {}
    for rec in summary:
        ep = rec["episode"]
        ff = np.asarray(Image.open(rec["first_frame"]).convert("RGB"))
        panels = []
        report[ep] = {}
        for cond, color in (("real", (60, 230, 60)), ("distractor", (255, 70, 70))):
            c = rec["conditions"][cond]
            panel, stats = overlay(ff, c["mask_png"], color, f"{ep[:28]} {cond}: {c['phrase']}")
            panels.append(panel)
            report[ep][cond] = {"phrase": c["phrase"], "blobs": stats}
            print(f"{ep} [{cond}] {c['phrase']!r}: {len(stats)} blobs "
                  f"{[b['px'] for b in stats]}")
        rows.append(np.concatenate(panels, axis=1))

    grid = np.concatenate(rows, axis=0)
    out = args.out or os.path.join(work, "mask_check_grid.jpg")
    im = Image.fromarray(grid)
    im.thumbnail((2000, 4000))
    im.save(out, quality=88)
    json.dump(report, open(os.path.join(work, "mask_check_report.json"), "w"), indent=2)
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
