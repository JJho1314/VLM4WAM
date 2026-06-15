#!/usr/bin/env python3
"""Full re-precompute with the fine-tuned InstructSAM: ONE inference pass per
video produces BOTH feature types plus an auditable mask record.

Outputs per video stem, into the dataset dir:
  <out-proj-name>/{stem}.pt    mask_query 256-d [SEG] projection  [L,256]
  <out-raw-name>/{stem}.pt     raw [SEG] hidden states            [L,2048]
  <out-proj-name>/{stem}_mask.png   best predicted mask (binary, for audit)

Both payloads carry query/phrase/caption/score/mask_png/mask_area so grounding
quality can be audited per-sample afterwards.

Sharding via RANK/WORLD_SIZE (torchrun --nproc_per_node=N). Runs in any env that
can run InstructSAM inference (no cosmos runtime needed beyond the light bridge;
pair with scripts/_env_stubs + COSMOS_SKIP_CUDA_VERSION_CHECK=1 if needed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_predict2._src.predict2.target_aware.instructsam_mask import InstructSAMTargetMaskGenerator

STOP_WORDS = {
    "after", "and", "before", "beside", "by", "from", "in", "inside", "into",
    "near", "next", "of", "on", "onto", "over", "then", "to", "under", "using", "with",
}
INVALID_TARGET_PREFIXES = STOP_WORDS | {
    "drop", "flip", "grab", "lift", "move", "pick", "pickup", "place", "pull",
    "push", "put", "remove", "slide", "take", "turn",
}


def _rank_info() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def load_caption(dataset_dir: Path, stem: str) -> str:
    text_path = dataset_dir / "metas" / f"{stem}.txt"
    if text_path.exists():
        return text_path.read_text().strip()
    raise FileNotFoundError(f"No caption for {stem} under {dataset_dir}/metas")


def extract_target_phrase(caption: str) -> str | None:
    if "[TGT]" not in caption:
        return None
    tail = caption.split("[TGT]", 1)[1].strip()
    tail = re.split(r"[,.;:!?]", tail, maxsplit=1)[0].strip()
    tokens = tail.split()
    first = tokens[0].strip("\"'`()[]{}").lower() if tokens else ""
    if first in INVALID_TARGET_PREFIXES:
        return None
    kept: list[str] = []
    for token in tokens:
        clean = token.strip("\"'`()[]{}").lower()
        if kept and clean in STOP_WORDS:
            break
        kept.append(token.strip("\"'`()[]{}"))
    return " ".join(p for p in kept if p) or None


def iter_videos(dataset_dir: Path, exclude_file: str) -> list[Path]:
    videos_dir = dataset_dir / "videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Missing videos dir: {videos_dir}")
    excluded: set[str] = set()
    if exclude_file != "none":
        p = dataset_dir / "exclude_no_tgt_stems.txt" if exclude_file == "auto" else Path(exclude_file)
        if p.exists():
            excluded = set(p.read_text().split())
    videos = sorted(v for v in videos_dir.glob("*.mp4") if v.stem not in excluded)
    if not videos:
        raise RuntimeError(f"No videos in {videos_dir}")
    return videos


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_payload(path: Path, rank: int, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".rank{rank}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", action="append", required=True)
    ap.add_argument("--model-path", required=True, help="Fine-tuned InstructSAM merged model.")
    ap.add_argument("--source-root", default=os.environ.get("INSTRUCTSAM_SOURCE_ROOT"))
    ap.add_argument("--out-proj-name", default="target_features_ft")
    ap.add_argument("--out-raw-name", default="target_features_rawseg_ft")
    ap.add_argument("--exclude-video-stems-file", default="auto")
    ap.add_argument("--query-template", default="Please segment '{target}' in the image.")
    ap.add_argument("--fallback-query", default="Please segment the target object in the image.")
    ap.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-errors", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    rank, local_rank, world = _rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map: str | dict = {"": f"cuda:{local_rank}"}
    else:
        device_map = "cpu"

    items: list[tuple[Path, Path]] = []
    for ds in args.dataset_dir:
        d = Path(ds)
        (d / args.out_proj_name).mkdir(parents=True, exist_ok=True)
        (d / args.out_raw_name).mkdir(parents=True, exist_ok=True)
        for v in iter_videos(d, args.exclude_video_stems_file):
            items.append((d, v))
    if args.limit > 0:
        items = items[: args.limit]
    shard = [it for i, it in enumerate(items) if i % world == rank]
    print(f"rank={rank}/{world} total={len(items)} shard={len(shard)} model={args.model_path}", flush=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    gen = InstructSAMTargetMaskGenerator(
        args.model_path, source_root=args.source_root, device_map=device_map, torch_dtype=dtype
    )

    errors = processed = skipped = 0
    t0 = time.time()
    for d, v in shard:
        proj_path = d / args.out_proj_name / f"{v.stem}.pt"
        raw_path = d / args.out_raw_name / f"{v.stem}.pt"
        summary = d / args.out_proj_name / f"precompute_ft_rank{rank:03d}.jsonl"
        if args.skip_existing and proj_path.exists() and raw_path.exists():
            skipped += 1
            continue
        caption = query = phrase = None
        try:
            caption = load_caption(d, v.stem)
            phrase = extract_target_phrase(caption)
            query = args.query_template.format(target=phrase) if phrase else args.fallback_query

            result = gen.predict_from_input(v, query, feature_mode="mask_query")
            if result.feature_B_L_D is None:
                raise RuntimeError("no seg_output_embeddings")
            proj_feat = result.feature_B_L_D.squeeze(0).float().contiguous()      # [L,256]
            # Same inference's seg embeddings, un-projected.
            raw = gen._extract_target_feature(feature_mode="raw_seg")
            raw_feat = raw.squeeze(0).float().contiguous() if raw is not None else None  # [L,2048]

            mask_png = None
            mask_area = None
            if result.mask_B_C_T_H_W is not None:
                m = (result.mask_B_C_T_H_W[0, 0, 0].numpy() > 0.5)
                mask_area = float(m.mean())
                mask_png = str(d / args.out_proj_name / f"{v.stem}_mask.png")
                Image.fromarray((m * 255).astype(np.uint8)).save(mask_png)

            meta = {
                "query": query, "target_phrase": phrase, "caption": caption,
                "instructsam_text": result.text, "score": result.score,
                "mask_png": mask_png, "mask_area_fraction": mask_area,
                "extractor": str(args.model_path),
            }
            save_payload(proj_path, rank, {"target_feature": proj_feat, "feature_mode": "mask_query_ft", **meta})
            if raw_feat is not None:
                save_payload(raw_path, rank, {"target_feature": raw_feat, "feature_mode": "raw_seg_ft", **meta})
            write_jsonl(summary, {"status": "ok", "stem": v.stem, "score": result.score,
                                  "mask_area": mask_area, "proj_shape": list(proj_feat.shape),
                                  "raw_shape": None if raw_feat is None else list(raw_feat.shape)})
            processed += 1
        except Exception as exc:
            errors += 1
            write_jsonl(summary, {"status": "error", "stem": v.stem, "error": repr(exc),
                                  "traceback": traceback.format_exc()})
            print(f"[rank {rank}] ERROR {v}: {exc}", file=sys.stderr, flush=True)
            if errors > args.max_errors:
                return 1
        if args.log_every and (processed + errors) % args.log_every == 0:
            el = max(time.time() - t0, 1e-6)
            print(f"rank={rank} done={processed} skip={skipped} err={errors} rate={processed/el:.3f}/s "
                  f"eta={((len(shard)-processed-skipped)/max(processed/el,1e-6))/3600:.1f}h", flush=True)

    print(f"rank={rank} FINISHED done={processed} skip={skipped} err={errors}", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
