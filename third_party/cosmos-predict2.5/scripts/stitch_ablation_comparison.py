#!/usr/bin/env python3
"""Stitch the target-query ablation into side-by-side comparisons.

For each scene, place the REAL-target generation next to the DISTRACTOR-target
generation (same conditioning first frame, different InstructSAM feature), with
query labels. Emits per-scene <ep>__compare.mp4 and a montage PNG (start/mid/end
frames, real on top, distractor on bottom).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import decord  # noqa: E402
decord.bridge.set_bridge("native")


def _read_video(path: str) -> np.ndarray:
    vr = decord.VideoReader(path, num_threads=2)
    return vr.get_batch(list(range(len(vr)))).asnumpy()  # [T,H,W,3] uint8


def _label(img: np.ndarray, text: str, color=(255, 230, 0)) -> np.ndarray:
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    # simple banner
    d.rectangle([0, 0, pil.width, 18], fill=(0, 0, 0))
    d.text((4, 4), text[:64], fill=color)
    return np.asarray(pil)


def _write_mp4(frames: np.ndarray, path: str, fps: int = 8) -> None:
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(path, list(frames), fps=fps, quality=8, macro_block_size=1)
    except Exception:
        import torch
        import torchvision
        torchvision.io.write_video(path, torch.from_numpy(frames), fps=fps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--features-dir", required=True, help="To read the real/distractor query labels.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _query_label(ep: str, cond: str) -> str:
        import torch
        fp = Path(args.features_dir) / f"{ep}__{cond}.pt"
        if fp.exists():
            q = torch.load(fp, map_location="cpu", weights_only=False).get("query", "")
            # "Please segment 'X' in the image." -> X
            if "'" in q:
                return q.split("'")[1]
            return q
        return cond

    reals = sorted(glob.glob(os.path.join(args.videos_dir, "*__real.mp4")))
    print(f"found {len(reals)} real videos")
    for rp in reals:
        ep = Path(rp).name[: -len("__real.mp4")]
        dp = os.path.join(args.videos_dir, f"{ep}__distractor.mp4")
        if not os.path.exists(dp):
            print(f"skip {ep}: no distractor video")
            continue
        rv, dv = _read_video(rp), _read_video(dp)
        T = min(len(rv), len(dv))
        rv, dv = rv[:T], dv[:T]
        rlab, dlab = _query_label(ep, "real"), _query_label(ep, "distractor")

        # side-by-side video (real | distractor)
        frames = []
        for t in range(T):
            left = _label(rv[t], f"REAL: {rlab}", (120, 255, 120))
            right = _label(dv[t], f"DISTRACTOR: {dlab}", (255, 140, 140))
            frames.append(np.concatenate([left, right], axis=1))
        comp_path = str(out / f"{ep}__compare.mp4")
        _write_mp4(np.stack(frames), comp_path, fps=args.fps)

        # montage PNG: rows = real/distractor, cols = start/mid/end
        idx = [0, T // 2, T - 1]
        row_r = np.concatenate([_label(rv[i], f"REAL t={i}", (120, 255, 120)) for i in idx], axis=1)
        row_d = np.concatenate([_label(dv[i], f"DISTRACTOR t={i}", (255, 140, 140)) for i in idx], axis=1)
        montage = np.concatenate([row_r, row_d], axis=0)
        png_path = str(out / f"{ep}__montage.png")
        Image.fromarray(montage).save(png_path)
        print(f"{ep}: real={rlab!r} distractor={dlab!r} -> {comp_path} + {png_path}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
