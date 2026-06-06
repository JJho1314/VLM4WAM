#!/usr/bin/env python3
"""Precompute InstructSAM target features for Cosmos target-aware training."""

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

import torch

from cosmos_predict2._src.predict2.target_aware.instructsam_mask import InstructSAMTargetMaskGenerator


STOP_WORDS = {
    "after",
    "and",
    "before",
    "beside",
    "by",
    "from",
    "in",
    "inside",
    "into",
    "near",
    "next",
    "of",
    "on",
    "onto",
    "over",
    "then",
    "to",
    "under",
    "using",
    "with",
}

INVALID_TARGET_PREFIXES = STOP_WORDS | {
    "drop",
    "flip",
    "grab",
    "lift",
    "move",
    "pick",
    "pickup",
    "place",
    "pull",
    "push",
    "put",
    "remove",
    "slide",
    "take",
    "turn",
}


def _default_source_root() -> Path:
    if os.environ.get("INSTRUCTSAM_SOURCE_ROOT"):
        return Path(os.environ["INSTRUCTSAM_SOURCE_ROOT"])
    return Path(__file__).resolve().parents[2] / "InstructSAM"


def _default_model_path() -> Path:
    if os.environ.get("INSTRUCTSAM_MODEL_PATH"):
        return Path(os.environ["INSTRUCTSAM_MODEL_PATH"])
    return _default_source_root() / "work_dirs" / "InstructSAM-2B"


def _rank_info() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    return rank, local_rank, world_size


def _load_json_caption(json_path: Path) -> str:
    with json_path.open("r") as f:
        content = f.read()
    data = json.loads(content if content.strip().startswith("{") else "{" + content + "}")
    first_model_value = next(iter(data.values()))
    if isinstance(first_model_value, dict):
        return str(next(iter(first_model_value.values())))
    return str(first_model_value)


def load_caption(dataset_dir: Path, stem: str) -> str:
    text_path = dataset_dir / "metas" / f"{stem}.txt"
    if text_path.exists():
        return text_path.read_text().strip()
    json_path = dataset_dir / "captions" / f"{stem}.json"
    if json_path.exists():
        return _load_json_caption(json_path).strip()
    raise FileNotFoundError(f"No caption found for {stem} under {dataset_dir}/metas or {dataset_dir}/captions")


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
    phrase = " ".join(part for part in kept if part).strip()
    return phrase or None


def build_query(caption: str, template: str, fallback_query: str) -> tuple[str, str | None]:
    phrase = extract_target_phrase(caption)
    if phrase is None:
        return fallback_query, None
    return template.format(target=phrase, caption=caption), phrase


def load_excluded_stems(dataset_dir: Path, exclude_file: str) -> set[str]:
    if exclude_file.lower() == "none":
        return set()
    path = dataset_dir / "exclude_no_tgt_stems.txt" if exclude_file == "auto" else Path(exclude_file)
    if not path.exists():
        return set()
    return set(path.read_text().split())


def iter_videos(dataset_dir: Path, exclude_file: str) -> list[Path]:
    videos_dir = dataset_dir / "videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Missing videos directory: {videos_dir}")
    excluded = load_excluded_stems(dataset_dir, exclude_file)
    videos = sorted(path for path in videos_dir.glob("*.mp4") if path.stem not in excluded)
    if not videos:
        raise RuntimeError(f"No active mp4 videos found in {videos_dir}")
    return videos


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_missing_instructsam_feature_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "did not return a mask or target feature" in text
        or "did not expose seg_output_embeddings" in text
    )


def zero_target_feature(num_tokens: int, feature_dim: int) -> torch.Tensor:
    return torch.zeros(max(1, int(num_tokens)), int(feature_dim), dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", action="append", required=True, help="Dataset root. Can be passed multiple times.")
    parser.add_argument("--model-path", type=Path, default=_default_model_path())
    parser.add_argument("--source-root", type=Path, default=_default_source_root())
    parser.add_argument("--output-dir-name", default="target_features")
    parser.add_argument("--exclude-video-stems-file", default="auto")
    parser.add_argument("--query-template", default="Please segment '{target}' in the image.")
    parser.add_argument("--fallback-query", default="Please segment the target object in the image.")
    parser.add_argument("--combine-mode", choices=["best", "union"], default="best")
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--feature-mode", choices=["mask_query", "raw_seg"], default="mask_query")
    parser.add_argument("--expected-feature-dim", type=int, default=256)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=0, help="Abort after this many failures. 0 means abort on the first failure.")
    parser.add_argument(
        "--fallback-zero-on-missing-feature",
        action="store_true",
        help="Write an all-zero feature when InstructSAM returns no seg_output_embeddings for a sample.",
    )
    parser.add_argument("--fallback-zero-tokens", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size = _rank_info()

    if not args.model_path.exists():
        raise FileNotFoundError(f"InstructSAM model path does not exist: {args.model_path}")
    if not args.source_root.exists():
        raise FileNotFoundError(f"InstructSAM source root does not exist: {args.source_root}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map: str | dict[str, str] = {"": f"cuda:{local_rank}"}
    else:
        device_map = "cpu"

    all_items: list[tuple[Path, Path, Path]] = []
    for dataset_dir_str in args.dataset_dir:
        dataset_dir = Path(dataset_dir_str)
        output_dir = dataset_dir / args.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for video_path in iter_videos(dataset_dir, args.exclude_video_stems_file):
            all_items.append((dataset_dir, output_dir, video_path))

    if args.limit > 0:
        all_items = all_items[: args.limit]
    shard_items = [item for idx, item in enumerate(all_items) if idx % world_size == rank]

    print(
        f"rank={rank} local_rank={local_rank} world_size={world_size} total_items={len(all_items)} "
        f"shard_items={len(shard_items)} model={args.model_path} source={args.source_root}",
        flush=True,
    )
    if args.dry_run:
        for dataset_dir, output_dir, video_path in shard_items[:10]:
            caption = load_caption(dataset_dir, video_path.stem)
            query, phrase = build_query(caption, args.query_template, args.fallback_query)
            print(f"DRYRUN {video_path} -> {output_dir / (video_path.stem + '.pt')} phrase={phrase!r} query={query!r}")
        return 0

    generator = InstructSAMTargetMaskGenerator(
        args.model_path,
        source_root=args.source_root,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
    )

    errors = 0
    zero_fallbacks = 0
    processed = 0
    skipped = 0
    start_time = time.time()
    for dataset_dir, output_dir, video_path in shard_items:
        output_path = output_dir / f"{video_path.stem}.pt"
        summary_path = output_dir / f"precompute_rank{rank:03d}.jsonl"
        if output_path.exists() and args.skip_existing and not args.overwrite:
            skipped += 1
            continue

        caption: str | None = None
        query: str | None = None
        phrase: str | None = None
        try:
            caption = load_caption(dataset_dir, video_path.stem)
            query, phrase = build_query(caption, args.query_template, args.fallback_query)
            result = generator.predict_from_input(
                video_path,
                query,
                combine_mode=args.combine_mode,
                mask_threshold=args.mask_threshold,
                feature_mode=args.feature_mode,
            )
            if result.feature_B_L_D is None:
                raise RuntimeError("InstructSAM did not expose seg_output_embeddings for target feature export")
            target_feature = result.feature_B_L_D.squeeze(0).detach().cpu().float().contiguous()
            if target_feature.ndim != 2:
                raise RuntimeError(f"Expected [L,D] target feature, got {tuple(target_feature.shape)}")
            if args.expected_feature_dim > 0 and target_feature.shape[-1] != args.expected_feature_dim:
                raise RuntimeError(
                    f"Expected feature dim {args.expected_feature_dim}, got {target_feature.shape[-1]}"
                )

            payload = {
                "target_feature": target_feature,
                "query": query,
                "target_phrase": phrase,
                "caption": caption,
                "instructsam_text": result.text,
                "score": result.score,
                "feature_mode": args.feature_mode,
            }
            tmp_path = output_path.with_suffix(output_path.suffix + f".rank{rank}.tmp")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, output_path)
            write_jsonl(
                summary_path,
                {
                    "status": "ok",
                    "stem": video_path.stem,
                    "query": query,
                    "target_phrase": phrase,
                    "score": result.score,
                    "feature_shape": list(target_feature.shape),
                },
            )
            processed += 1
        except Exception as exc:
            if args.fallback_zero_on_missing_feature and is_missing_instructsam_feature_error(exc):
                target_feature = zero_target_feature(args.fallback_zero_tokens, args.expected_feature_dim)
                payload = {
                    "target_feature": target_feature,
                    "query": query,
                    "target_phrase": phrase,
                    "caption": caption,
                    "instructsam_text": None,
                    "score": None,
                    "feature_mode": args.feature_mode,
                    "fallback_zero": True,
                    "fallback_error": repr(exc),
                }
                tmp_path = output_path.with_suffix(output_path.suffix + f".rank{rank}.tmp")
                torch.save(payload, tmp_path)
                os.replace(tmp_path, output_path)
                write_jsonl(
                    summary_path,
                    {
                        "status": "fallback_zero",
                        "stem": video_path.stem,
                        "query": query,
                        "target_phrase": phrase,
                        "error": repr(exc),
                        "feature_shape": list(target_feature.shape),
                    },
                )
                zero_fallbacks += 1
                processed += 1
                print(
                    f"[rank {rank}] FALLBACK_ZERO {video_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            errors += 1
            write_jsonl(
                summary_path,
                {
                    "status": "error",
                    "stem": video_path.stem,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[rank {rank}] ERROR {video_path}: {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            if args.max_errors == 0 or errors > args.max_errors:
                return 1

        if args.log_every > 0 and (processed + errors) % args.log_every == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            rate = processed / elapsed
            print(
                f"rank={rank} processed={processed} skipped={skipped} zero_fallbacks={zero_fallbacks} errors={errors} "
                f"rate={rate:.3f}/s elapsed={elapsed/3600:.2f}h",
                flush=True,
            )

    print(
        f"rank={rank} done processed={processed} skipped={skipped} zero_fallbacks={zero_fallbacks} errors={errors}",
        flush=True,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
