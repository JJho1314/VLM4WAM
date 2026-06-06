"""Link one or more Cosmos VideoDataset directories into one directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def link_pair(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "metas").mkdir(parents=True, exist_ok=True)

    count = 0
    source_counts = {}
    for source in args.sources:
        src = Path(source).resolve()
        videos = sorted((src / "videos").glob("*.mp4"))
        source_counts[str(src)] = len(videos)
        for video in videos:
            name = video.stem
            meta = src / "metas" / f"{name}.txt"
            if not meta.exists():
                raise RuntimeError(f"Missing caption for {video}: {meta}")
            link_pair(video, out / "videos" / video.name)
            link_pair(meta, out / "metas" / meta.name)
            count += 1

    summary = {"out": str(out), "total": count, "sources": source_counts}
    (out / "link_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
