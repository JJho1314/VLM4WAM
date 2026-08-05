"""Atomically migrate completed WorldArena Baton head checkpoints to metadata v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from qwen35_baton.checkpoint import migrate_legacy_head_checkpoint_v4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results = [
        migrate_legacy_head_checkpoint_v4(checkpoint).to_dict()
        for checkpoint in args.checkpoints
    ]
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
