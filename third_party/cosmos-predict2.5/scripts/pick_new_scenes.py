#!/usr/bin/env python3
"""Build a fresh-scene eval batch for the current cosmos + (FT) InstructSAM combo.

Picks N unused episodes from a holdout dataset (diverse captions), extracts
first frames, and emits:
  - manifest.json  : stage1 feature extraction (one real-target query per scene)
  - samples.jsonl  : stage2 generation (real-target + zero per scene, image2world)
  - summary.json   : per-scene record (caption / phrase / paths)
Run with the cosmos venv python (needs torch + decord + PIL).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

STOP = {"after", "and", "before", "beside", "by", "from", "in", "inside", "into",
        "near", "next", "of", "on", "onto", "over", "then", "to", "under", "using", "with"}
STRIP = "\"'`()[]{}"


def tgt_phrase(caption: str) -> str | None:
    if "[TGT]" not in caption:
        return None
    tail = caption.split("[TGT]", 1)[1].strip()
    tail = re.split(r"[,.;:!?]", tail, maxsplit=1)[0].strip()
    kept: list[str] = []
    for tok in tail.split():
        clean = tok.strip(STRIP).lower()
        if kept and clean in STOP:
            break
        kept.append(tok.strip(STRIP))
    return " ".join(kept) or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--query-style", choices=["grasp", "segment"], default="grasp")
    ap.add_argument("--num-frames", type=int, default=49)
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=20260526)
    args = ap.parse_args()

    import decord
    from PIL import Image
    decord.bridge.set_bridge("native")

    val = Path(args.val_dir)
    work = Path(args.work_dir)
    feat = work / "features"
    inputs = work / "inputs"
    feat.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    excluded = set(args.exclude)

    # Walk metas; pick scenes with distinct captions (task diversity), skipping
    # excluded episodes and episodes without a parseable [TGT] phrase.
    picked = []
    seen_caps = set()
    for meta in sorted((val / "metas").glob("*.txt")):
        ep = meta.stem
        if ep in excluded:
            continue
        caption = meta.read_text().strip()
        phrase = tgt_phrase(caption)
        if not phrase:
            continue
        cap_key = caption.split("[TGT]", 1)[1][:60]
        if cap_key in seen_caps:
            continue
        if not (val / "videos" / f"{ep}.mp4").exists():
            continue
        seen_caps.add(cap_key)
        picked.append((ep, caption, phrase))
        if len(picked) >= args.num:
            break

    # Shared zero feature for the unguided control.
    zero_path = feat / "zero.pt"
    torch.save({"target_feature": torch.zeros(64, 256), "query": "<zero>"}, zero_path)

    manifest, samples, summary = [], [], []
    for ep, caption, phrase in picked:
        video = val / "videos" / f"{ep}.mp4"
        ff = inputs / f"{ep}__first_frame.png"
        if not ff.exists():
            vr = decord.VideoReader(str(video), num_threads=2)
            Image.fromarray(vr.get_batch([0]).asnumpy()[0]).save(ff)
        (inputs / f"{ep}__caption.txt").write_text(caption + "\n")
        if args.query_style == "grasp":
            query = f"Grasp {phrase}."
        else:
            query = f"Please segment '{phrase}' in the image."
        out_feat = feat / f"{ep}__real.pt"
        manifest.append({"image": str(video), "query": query, "out": str(out_feat)})
        prompt = caption.replace("[TGT] ", "").replace("[TGT]", "").strip()
        common = {"inference_type": "image2world", "input_path": str(ff), "prompt": prompt,
                  "num_output_frames": args.num_frames, "num_steps": args.num_steps,
                  "guidance": args.guidance, "seed": args.seed}
        samples.append({"name": f"{ep}__real", "target_feature_path": str(out_feat), **common})
        samples.append({"name": f"{ep}__zero", "target_feature_path": str(zero_path), **common})
        summary.append({"episode": ep, "caption": caption, "phrase": phrase, "query": query,
                        "gt_video": str(video), "first_frame": str(ff)})

    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (work / "samples.jsonl").open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"scenes={len(picked)}  stage1={len(manifest)}  stage2={len(samples)}")
    for ep, _, phrase in picked:
        print(f"  {ep}: target={phrase!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
