#!/usr/bin/env python3
"""Precompute fused multi-source InstructSAM target features for text-free Cosmos.

For each video, run InstructSAM once and fuse three representations
(mask / detect / vtext) into a SINGLE ``[L, 256]`` ``target_feature`` tensor with
fixed per-source token budgets, stored exactly like the single-feature pipeline
so the existing ``VideoDataset`` loader is reused unchanged.

Layout (fixed, deterministic order):  ``[ mask | detect | vtext ]``
Default budgets: mask 16 / detect 16 / vtext 32  -> 64 tokens, dim 256.

``vtext`` (native 4096-d) is reduced to 256-d by a fixed seeded orthonormal
projection saved next to the features so it is identical across shards /
train / val.
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
from typing import Any, Optional

import torch

from cosmos_predict2._src.predict2.target_aware.instructsam_multisource import (
    InstructSAMMultiSourceGenerator,
    MultiSourceFeatureResult,
)


STOP_WORDS = {
    "after", "and", "before", "beside", "by", "from", "in", "inside", "into",
    "near", "next", "of", "on", "onto", "over", "then", "to", "under", "using", "with",
}
INVALID_TARGET_PREFIXES = STOP_WORDS | {
    "drop", "flip", "grab", "lift", "move", "pick", "pickup", "place", "pull",
    "push", "put", "remove", "slide", "take", "turn",
}

SOURCES = ("mask", "detect", "vtext")


# --------------------------------------------------------------------------- #
# Helpers shared with precompute_instructsam_target_features.py (kept local to
# avoid cross-script imports).
# --------------------------------------------------------------------------- #
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


def extract_target_phrase(caption: str) -> Optional[str]:
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


def build_query(caption: str, template: str, fallback_query: str) -> tuple[str, Optional[str]]:
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
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Fusion.
# --------------------------------------------------------------------------- #
class SourceProjector:
    """Project any source rep ``[L, native]`` -> ``[L, out_dim]``.

    Identity when ``native == out_dim`` (so mask/detect stay as their learned
    256-d features), otherwise a FIXED seeded projection (orthonormal columns
    when native >= out_dim, scaled Gaussian otherwise). Projections are sized
    lazily from the data — NO hardcoded native dims — so a source is never
    silently zeroed because its real dim (e.g. vtext = 2048, not 4096) differs
    from a guess. Deterministic from ``(seed, native, out)`` and persisted, so
    every rank / re-run agrees.
    """

    def __init__(self, proj_dir: Path, out_dim: int, seed: int) -> None:
        self.proj_dir = Path(proj_dir)
        self.out_dim = int(out_dim)
        self.seed = int(seed)
        self._cache: dict[int, Optional[torch.Tensor]] = {}

    def _matrix(self, native_dim: int) -> Optional[torch.Tensor]:
        if native_dim == self.out_dim:
            return None  # identity
        if native_dim in self._cache:
            return self._cache[native_dim]
        path = self.proj_dir / f"_proj_{native_dim}x{self.out_dim}_seed{self.seed}.pt"
        proj: Optional[torch.Tensor] = None
        if path.exists():
            cand = torch.load(path, map_location="cpu")
            if tuple(cand.shape) == (native_dim, self.out_dim):
                proj = cand.float()
        if proj is None:
            gen = torch.Generator().manual_seed(self.seed * 1_000_003 + native_dim)
            mat = torch.randn(native_dim, self.out_dim, generator=gen)
            if native_dim >= self.out_dim:
                q, _ = torch.linalg.qr(mat, mode="reduced")  # [native, out] orthonormal cols
                proj = q[:, : self.out_dim].contiguous().float()
            else:
                proj = (mat / (native_dim ** 0.5)).contiguous().float()
            self.proj_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")  # pid-unique: no rank race
            torch.save(proj, tmp)
            os.replace(tmp, path)
        self._cache[native_dim] = proj
        return proj

    def project(self, rep: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if rep is None or rep.numel() == 0 or rep.ndim != 2:
            return None
        rep = rep.detach().float().cpu()
        proj = self._matrix(rep.shape[-1])
        return rep if proj is None else (rep @ proj)


def _fit_budget(rep: Optional[torch.Tensor], budget: int, dim: int) -> torch.Tensor:
    """Truncate/pad a ``[L, dim]`` rep (or None) to exactly ``[budget, dim]``."""
    out = torch.zeros(budget, dim, dtype=torch.float32)
    if rep is None or rep.numel() == 0:
        return out
    rep = rep.detach().float().cpu()
    if rep.ndim != 2 or rep.shape[-1] != dim:
        return out
    n = min(rep.shape[0], budget)
    out[:n] = rep[:n]
    return out


def fuse(
    result: MultiSourceFeatureResult,
    projector: "SourceProjector",
    budgets: dict[str, int],
    out_dim: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Return fused ``[sum(budgets), out_dim]`` tensor + per-source segment sizes.

    Every source is projected to ``out_dim`` first, so a dim mismatch can never
    silently zero a source.
    """
    mask = _fit_budget(projector.project(result.mask_L_Dm), budgets["mask"], out_dim)
    detect = _fit_budget(projector.project(result.detect_L_Dd), budgets["detect"], out_dim)
    vtext = _fit_budget(projector.project(result.vtext_L_Dv), budgets["vtext"], out_dim)
    fused = torch.cat([mask, detect, vtext], dim=0).contiguous()
    segments = {s: budgets[s] for s in SOURCES}
    return fused, segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", action="append", required=True, help="Dataset root (repeatable).")
    parser.add_argument("--model-path", type=Path, default=_default_model_path())
    parser.add_argument("--source-root", type=Path, default=_default_source_root())
    parser.add_argument("--output-dir-name", default="target_features_multisource")
    parser.add_argument("--exclude-video-stems-file", default="auto")
    parser.add_argument("--query-template", default="Please segment '{target}' in the image.")
    parser.add_argument("--fallback-query", default="Please segment the target object in the image.")
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--mask-tokens", type=int, default=16)
    parser.add_argument("--detect-tokens", type=int, default=16)
    parser.add_argument("--vtext-tokens", type=int, default=32)
    # Per-source native dims are auto-detected; sources whose dim != out-dim get a
    # fixed seeded projection (vtext is 2048-d for InstructSAM-2B, not 4096).
    parser.add_argument("--proj-seed", type=int, default=0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-errors", type=int, default=0, help="Abort after N failures (0 = abort on first).")
    parser.add_argument("--fallback-zero-on-missing-feature", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-shapes", action="store_true", help="Print native rep shapes for the first few items.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size = _rank_info()
    budgets = {"mask": args.mask_tokens, "detect": args.detect_tokens, "vtext": args.vtext_tokens}
    total_tokens = sum(budgets.values())

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

    # Per-source projections live in the FIRST dataset's output dir; deterministic
    # from the seed so every shard / dataset agrees.
    proj_dir = Path(args.dataset_dir[0]) / args.output_dir_name
    projector = SourceProjector(proj_dir, args.out_dim, args.proj_seed)

    print(
        f"rank={rank} local_rank={local_rank} world_size={world_size} total_items={len(all_items)} "
        f"shard_items={len(shard_items)} budgets={budgets} total_tokens={total_tokens} "
        f"model={args.model_path} proj_dir={proj_dir}",
        flush=True,
    )
    if args.dry_run:
        for dataset_dir, output_dir, video_path in shard_items[:10]:
            caption = load_caption(dataset_dir, video_path.stem)
            query, phrase = build_query(caption, args.query_template, args.fallback_query)
            print(f"DRYRUN {video_path} -> {output_dir / (video_path.stem + '.pt')} phrase={phrase!r} query={query!r}")
        return 0

    generator = InstructSAMMultiSourceGenerator(
        args.model_path,
        source_root=args.source_root,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
        detect_max_tokens=max(args.detect_tokens * 4, 64),
        vtext_max_tokens=max(args.vtext_tokens * 4, 64),
    )

    errors = processed = skipped = zero_fallbacks = 0
    start_time = time.time()
    for idx, (dataset_dir, output_dir, video_path) in enumerate(shard_items):
        output_path = output_dir / f"{video_path.stem}.pt"
        summary_path = output_dir / f"precompute_rank{rank:03d}.jsonl"
        if output_path.exists() and args.skip_existing and not args.overwrite:
            skipped += 1
            continue

        caption = query = phrase = None
        try:
            caption = load_caption(dataset_dir, video_path.stem)
            query, phrase = build_query(caption, args.query_template, args.fallback_query)
            result = generator.predict_multi_source_from_input(video_path, query)

            if args.debug_shapes and idx < 5:
                def _sh(t):
                    return None if t is None else list(t.shape)
                print(
                    f"[rank {rank}] SHAPES {video_path.stem} mask={_sh(result.mask_L_Dm)} "
                    f"detect={_sh(result.detect_L_Dd)} vtext={_sh(result.vtext_L_Dv)}",
                    flush=True,
                )

            if not result.any_present():
                raise RuntimeError("InstructSAM did not expose any target feature (mask/detect/vtext) for export")

            fused, segments = fuse(result, projector, budgets, args.out_dim)
            if fused.shape != (total_tokens, args.out_dim):
                raise RuntimeError(f"Fused feature shape {tuple(fused.shape)} != {(total_tokens, args.out_dim)}")

            payload = {
                "target_feature": fused,            # [total_tokens, out_dim] -- dataloader key
                "source_segments": segments,        # {"mask":Nm,"detect":Nd,"vtext":Nv}
                "source_order": list(SOURCES),
                "query": query,
                "target_phrase": phrase,
                "caption": caption,
                "instructsam_text": result.text,
                "score": result.score,
                "feature_mode": "multisource",
            }
            tmp_path = output_path.with_suffix(output_path.suffix + f".rank{rank}.tmp")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, output_path)
            write_jsonl(summary_path, {
                "status": "ok", "stem": video_path.stem, "query": query, "target_phrase": phrase,
                "score": result.score, "feature_shape": list(fused.shape), "segments": segments,
            })
            processed += 1
        except Exception as exc:
            is_missing = "did not expose any target feature" in str(exc)
            if args.fallback_zero_on_missing_feature and is_missing:
                fused = torch.zeros(total_tokens, args.out_dim, dtype=torch.float32)
                payload = {
                    "target_feature": fused,
                    "source_segments": budgets,
                    "source_order": list(SOURCES),
                    "query": query, "target_phrase": phrase, "caption": caption,
                    "instructsam_text": None, "score": None, "feature_mode": "multisource",
                    "fallback_zero": True, "fallback_error": repr(exc),
                }
                tmp_path = output_path.with_suffix(output_path.suffix + f".rank{rank}.tmp")
                torch.save(payload, tmp_path)
                os.replace(tmp_path, output_path)
                write_jsonl(summary_path, {
                    "status": "fallback_zero", "stem": video_path.stem, "error": repr(exc),
                    "feature_shape": list(fused.shape),
                })
                zero_fallbacks += 1
                processed += 1
                print(f"[rank {rank}] FALLBACK_ZERO {video_path}: {exc}", file=sys.stderr, flush=True)
                continue

            errors += 1
            write_jsonl(summary_path, {
                "status": "error", "stem": video_path.stem, "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"[rank {rank}] ERROR {video_path}: {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            if args.max_errors == 0 or errors > args.max_errors:
                return 1

        if args.log_every > 0 and (processed + errors) % args.log_every == 0:
            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"rank={rank} processed={processed} skipped={skipped} zero_fallbacks={zero_fallbacks} "
                f"errors={errors} rate={processed/elapsed:.3f}/s elapsed={elapsed/3600:.2f}h",
                flush=True,
            )

    print(
        f"rank={rank} done processed={processed} skipped={skipped} zero_fallbacks={zero_fallbacks} errors={errors}",
        flush=True,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
