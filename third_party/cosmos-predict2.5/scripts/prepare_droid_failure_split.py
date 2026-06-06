"""Create a deterministic train/test episode split for LeRobot DROID failure data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/data/user/jhe724/workspace/data/droid_failure")
    parser.add_argument("--out", default="/data/user/jhe724/workspace/datasets/droid_failure_split")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--min-frames", type=int, default=33)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for chunk_dir in sorted((src / "meta" / "episodes").iterdir()):
        for parquet_path in sorted(chunk_dir.glob("*.parquet")):
            rows.append(pd.read_parquet(parquet_path))
    episodes = pd.concat(rows, ignore_index=True)
    eligible = [int(v) for v in episodes.loc[episodes["length"] >= args.min_frames, "episode_index"].tolist()]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)

    test_count = max(1, round(len(eligible) * args.test_ratio))
    test_ids = sorted(eligible[:test_count])
    train_ids = sorted(eligible[test_count:])

    (out / "train_episodes.txt").write_text("\n".join(map(str, train_ids)) + "\n")
    (out / "test_episodes.txt").write_text("\n".join(map(str, test_ids)) + "\n")
    summary = {
        "src": str(src),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "min_frames": args.min_frames,
        "total_rows": int(len(episodes)),
        "eligible_rows": len(eligible),
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
