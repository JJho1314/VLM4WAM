"""Predecode WorldArena training MP4s into deterministic per-episode HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from qwen35_baton.worldarena_data import (
    load_worldarena_source_manifest,
    predecode_worldarena,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predecode the WorldArena training-only release to HDF5."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Decode only the first N source records (smoke tests only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be a positive integer")
    dataset_root = args.dataset_root.expanduser().resolve()
    source_manifest = dataset_root / "metadata_train_a2v.jsonl"
    records = load_worldarena_source_manifest(source_manifest, dataset_root)
    if args.limit is not None:
        records = records[: args.limit]
    published = predecode_worldarena(
        records,
        output_root=args.output_root,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
