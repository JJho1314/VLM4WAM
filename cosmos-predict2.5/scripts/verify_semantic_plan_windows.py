#!/usr/bin/env python3
"""Pre-flight verification of semantic-plan window manifests before Cosmos training.

Catches the silent failure modes of the window pipeline:
- clamped / frozen tail frames (stale frame_ranges.json vs. actual video length shows up
  as repeated indices at the end of video_frame_indices),
- non-uniform frame stride inside a clip,
- keyframes outside the clip window, duplicate keyframes, non-monotonic keyframe times,
- missing or malformed semantic-plan .pt files,
- VAE-latent cache coverage gaps (records silently fall back to on-the-fly encoding).

Usage (from the cosmos-predict2.5 repo root on HPC3):
    python scripts/verify_semantic_plan_windows.py \
        --semantic-plan-dir $DATASET_ROOT/siglip2_semantic_plan_k16_g9_cosmos_t93_s123_step24_full \
        [--manifest 'manifest*.jsonl'] [--dataset-root $DATASET_ROOT] \
        [--check-payloads 64] [--check-video-lengths] [--vae-latent-dir DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-plan-dir", type=Path, required=True)
    parser.add_argument("--manifest", default="manifest*.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Needed for --check-video-lengths")
    parser.add_argument("--sequence-length", type=int, default=93)
    parser.add_argument("--check-payloads", type=int, default=32, help="Load N evenly spaced .pt payloads")
    parser.add_argument("--check-video-lengths", action="store_true", help="Read video lengths with decord")
    parser.add_argument("--vae-latent-dir", type=Path, default=None)
    parser.add_argument("--vae-latent-manifest", default="window_manifest*.jsonl")
    parser.add_argument("--max-report", type=int, default=20, help="Max offending records printed per issue")
    return parser.parse_args()


def load_manifest_records(directory: Path, pattern: str) -> list[dict]:
    paths = sorted(glob.glob(str(directory / pattern)))
    if not paths:
        raise FileNotFoundError(f"No manifest matching {directory}/{pattern}")
    records: list[dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    print(f"loaded {len(records)} records from {len(paths)} manifest file(s)")
    return records


def main() -> None:
    args = parse_args()
    records = load_manifest_records(args.semantic_plan_dir, args.manifest)

    issues: dict[str, list[str]] = defaultdict(list)
    stride_counter: Counter = Counter()
    windows_per_stem: Counter = Counter()
    missing_times_fields = 0

    for rec in records:
        sid = str(rec.get("sample_id", "<missing sample_id>"))
        clip = rec.get("video_frame_indices") or []
        future = rec.get("future_frame_indices") or []
        stride = int(rec.get("frame_stride") or 0)
        stride_counter[stride] += 1
        windows_per_stem[str(rec.get("stem", "?"))] += 1

        if not clip or not future:
            missing_times_fields += 1
            issues["missing frame-index fields (keyframe times fall back to uniform)"].append(sid)
            continue
        if len(clip) != args.sequence_length:
            issues[f"clip length != {args.sequence_length}"].append(f"{sid} (len={len(clip)})")
        diffs = [b - a for a, b in zip(clip[:-1], clip[1:])]
        if any(d <= 0 for d in diffs):
            issues["clip indices not strictly increasing (clamped/frozen tail frames)"].append(sid)
        elif stride > 0 and any(d != stride for d in diffs):
            issues["non-uniform stride inside clip"].append(f"{sid} (stride={stride}, diffs={sorted(set(diffs))})")
        if future[0] <= clip[0]:
            issues["first keyframe at/before conditioning frame"].append(sid)
        if future[-1] != clip[-1]:
            issues["last keyframe != last clip frame"].append(f"{sid} ({future[-1]} vs {clip[-1]})")
        if any(f < clip[0] or f > clip[-1] for f in future):
            issues["keyframe outside clip window"].append(sid)
        if any(b < a for a, b in zip(future[:-1], future[1:])):
            issues["keyframes not monotonic"].append(sid)
        if len(set(future)) != len(future):
            issues["duplicate keyframes (degenerate plan)"].append(sid)

    # Payload spot checks.
    if args.check_payloads > 0 and records:
        step = max(len(records) // args.check_payloads, 1)
        checked = 0
        for rec in records[::step][: args.check_payloads]:
            sid = str(rec["sample_id"])
            path = rec.get("path") or str(args.semantic_plan_dir / f"{sid}.pt")
            if not os.path.exists(path):
                issues["semantic-plan .pt missing"].append(sid)
                continue
            payload = torch.load(path, map_location="cpu", weights_only=False)
            plan = payload["semantic_plan"] if isinstance(payload, dict) else payload
            plan = torch.as_tensor(plan).float()
            expected_shape = rec.get("shape")
            if expected_shape and list(plan.shape) != list(expected_shape):
                issues["payload shape != manifest shape"].append(f"{sid} ({list(plan.shape)} vs {expected_shape})")
            if not torch.isfinite(plan).all():
                issues["payload has non-finite values"].append(sid)
            if plan.reshape(plan.shape[0], -1).abs().sum(dim=-1).eq(0).any():
                issues["payload has all-zero keyframes (treated as padding by the adapter)"].append(sid)
            checked += 1
        print(f"payload spot check: {checked} files loaded")

    # Video length check (catches stale frame_ranges.json even without clamped indices).
    if args.check_video_lengths:
        if args.dataset_root is None:
            raise ValueError("--dataset-root is required with --check-video-lengths")
        import decord

        lengths: dict[str, int] = {}
        for rec in records:
            stem = str(rec.get("stem", ""))
            clip = rec.get("video_frame_indices") or []
            if not stem or not clip:
                continue
            if stem not in lengths:
                video_path = args.dataset_root / "videos" / f"{stem}.mp4"
                if not video_path.exists():
                    issues["video file missing"].append(stem)
                    lengths[stem] = -1
                    continue
                lengths[stem] = len(decord.VideoReader(str(video_path), ctx=decord.cpu(0)))
            if lengths[stem] >= 0 and clip[-1] >= lengths[stem]:
                issues["clip exceeds video length (stale frame_ranges?)"].append(
                    f"{rec['sample_id']} (last={clip[-1]}, video_len={lengths[stem]})"
                )
        print(f"video length check: {len(lengths)} videos probed")

    # VAE latent coverage.
    if args.vae_latent_dir is not None:
        latent_ids: set[str] = set()
        for path in sorted(glob.glob(str(args.vae_latent_dir / args.vae_latent_manifest))):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        latent_ids.add(str(json.loads(line).get("sample_id")))
        covered = sum(1 for rec in records if str(rec["sample_id"]) in latent_ids)
        print(f"VAE latent coverage: {covered}/{len(records)} ({covered / max(len(records), 1):.1%})")
        if covered < len(records):
            issues["records without cached VAE latent (silently encode on the fly)"].append(
                f"{len(records) - covered} records"
            )

    # Summary.
    print("\n=== window statistics ===")
    print(f"total windows: {len(records)}, stems: {len(windows_per_stem)}")
    print(f"windows per stride: {dict(sorted(stride_counter.items()))}")
    counts = sorted(windows_per_stem.values())
    if counts:
        print(
            f"windows per stem: min={counts[0]} median={counts[len(counts) // 2]} max={counts[-1]}"
        )
    if missing_times_fields:
        print(
            f"NOTE: {missing_times_fields} records lack frame-index fields; their keyframe times "
            "fall back to the uniform-spacing assumption (pre-P4 behavior)."
        )

    print("\n=== issues ===")
    if not issues:
        print("none found")
        return
    for name, offenders in sorted(issues.items(), key=lambda kv: -len(kv[1])):
        print(f"[{len(offenders)}] {name}")
        for offender in offenders[: args.max_report]:
            print(f"    {offender}")
        if len(offenders) > args.max_report:
            print(f"    ... and {len(offenders) - args.max_report} more")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
