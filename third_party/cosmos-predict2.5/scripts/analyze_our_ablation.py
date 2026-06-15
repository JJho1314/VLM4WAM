#!/usr/bin/env python3
"""Real-vs-distractor diff analysis for the text-free multisource model ablation.

For each scene: 3-row montage (real / distractor / |distractor-real|x5) + the
mean abs RGB difference between the real-target and distractor-target generation.
A large, localized diff means changing the InstructSAM target steers our model.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import decord
import numpy as np
import torch
from PIL import Image, ImageDraw

decord.bridge.set_bridge("native")


def query_of(feat_dir: str, ep: str, cond: str) -> str:
    p = os.path.join(feat_dir, f"{ep}__{cond}.pt")
    if os.path.exists(p):
        s = torch.load(p, map_location="cpu", weights_only=False).get("query", "")
        if "'" in s:
            return s.split("'")[1]
        return s or cond
    return cond


def sample_frames(path: str, n: int = 12):
    # Sample a strided subset only (never load the full clip -> avoids login-node OOM).
    vr = decord.VideoReader(path, num_threads=2)
    T = len(vr)
    idx = [round(i * (T - 1) / (n - 1)) for i in range(n)]
    return vr.get_batch(idx).asnumpy(), idx, T


def label(img: np.ndarray, text: str, color) -> np.ndarray:
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    d.text((3, 2), text[:44], fill=color)
    return np.asarray(im)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--amplify", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    metrics = {}
    for rp in sorted(glob.glob(os.path.join(args.videos_dir, "*__real.mp4"))):
        ep = os.path.basename(rp)[: -len("__real.mp4")]
        dp = os.path.join(args.videos_dir, f"{ep}__distractor.mp4")
        if not os.path.exists(dp):
            continue
        rfN, idx, T = sample_frames(rp, 12)
        dfN, _, _ = sample_frames(dp, 12)
        n = min(len(rfN), len(dfN))
        mad = float(np.abs(rfN[:n].astype(np.float32) - dfN[:n].astype(np.float32)).mean())
        rq, dq = query_of(args.features_dir, ep, "real"), query_of(args.features_dir, ep, "distractor")
        metrics[ep] = {"real_q": rq, "distractor_q": dq, "mean_abs_rgb_real_vs_distractor": round(mad, 3)}

        mi = list(range(0, 12, 2))  # 6 frames for the montage
        rf6 = rfN[mi]; df6 = dfN[mi]; mtimes = [idx[j] for j in mi]
        diff = np.clip(np.abs(rf6.astype(np.int16) - df6.astype(np.int16)) * args.amplify, 0, 255).astype(np.uint8)
        row_r = np.concatenate([label(rf6[i], f"REAL:{rq} t={mtimes[i]}", (120, 255, 120)) for i in range(6)], axis=1)
        row_d = np.concatenate([label(df6[i], f"DIST:{dq} t={mtimes[i]}", (255, 150, 150)) for i in range(6)], axis=1)
        row_x = np.concatenate([label(diff[i], f"DIFFx{args.amplify} t={mtimes[i]}", (255, 255, 0)) for i in range(6)], axis=1)
        Image.fromarray(np.concatenate([row_r, row_d, row_x], axis=0)).save(
            os.path.join(args.out_dir, f"{ep}__ablation3row.png")
        )
        print(f"{ep}: real={rq!r} dist={dq!r} mean_abs_rgb={mad:.3f}", flush=True)

    json.dump(metrics, open(os.path.join(args.out_dir, "our_model_metrics.json"), "w"), indent=2)
    if metrics:
        vals = [m["mean_abs_rgb_real_vs_distractor"] for m in metrics.values()]
        print(f"mean over {len(metrics)} scenes: {round(float(np.mean(vals)), 3)}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
