"""Audit LIBERO task JSONL files for the libero_tgt_v1 contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from qwen3_vl_semantic_planner.libero_target_text import (
    LIBERO_TGT_PREPROCESSING,
    mark_libero_target,
)


def audit_task_files(paths: Sequence[Path]) -> dict[str, object]:
    if not paths:
        raise ValueError("at least one tasks.jsonl path is required")
    files: list[dict[str, object]] = []
    total = 0
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing LIBERO task metadata: {path}")
        count = 0
        examples: list[dict[str, str]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task")
            if not isinstance(task, str):
                raise ValueError(
                    f"{path}:{line_number} needs a string task field"
                )
            marked = mark_libero_target(task)
            count += 1
            if len(examples) < 3:
                examples.append({"raw": task, "marked": marked})
        if count == 0:
            raise ValueError(f"no tasks found in {path}")
        files.append(
            {
                "path": str(path),
                "tasks": count,
                "marked": count,
                "examples": examples,
            }
        )
        total += count
    return {
        "instruction_preprocessing": LIBERO_TGT_PREPROCESSING,
        "total_tasks": total,
        "total_marked": total,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_files", nargs="+", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_task_files(args.task_files),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
