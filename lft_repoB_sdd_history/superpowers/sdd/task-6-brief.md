### Task 6: Four-Suite Caption Audit and Training Preflight

**Files:**
- Create: `qwen3_vl_semantic_planner/audit_libero_target_text.py`
- Create: `tests/fixtures/libero_task_texts.json`
- Modify: `tests/test_libero_target_text.py`
- Modify: `qwen3_vl_semantic_planner/README.md`

**Interfaces:**
- Consumes: JSONL rows with a non-empty `task` string and `mark_libero_target`.
- Produces: `audit_task_files(paths: Sequence[Path]) -> dict[str, object]` and a read-only CLI that exits nonzero on the first incompatible instruction.

- [ ] **Step 1: Add the canonical four-suite fixture**

Create `tests/fixtures/libero_task_texts.json` with four keys and the exact 10 strings currently present in each HPC3 `meta/tasks.jsonl`:

```json
{
  "libero_10": [
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the yellow and white mug in the microwave and close it",
    "put both moka pots on the stove",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "pick up the book and place it in the back compartment of the caddy"
  ],
  "libero_goal": [
    "open the middle drawer of the cabinet",
    "open the top drawer and put the bowl inside",
    "push the plate to the front of the stove",
    "put the bowl on the plate",
    "put the bowl on the stove",
    "put the bowl on top of the cabinet",
    "put the cream cheese in the bowl",
    "put the wine bottle on the rack",
    "put the wine bottle on top of the cabinet",
    "turn on the stove"
  ],
  "libero_object": [
    "pick up the alphabet soup and place it in the basket",
    "pick up the bbq sauce and place it in the basket",
    "pick up the butter and place it in the basket",
    "pick up the chocolate pudding and place it in the basket",
    "pick up the cream cheese and place it in the basket",
    "pick up the ketchup and place it in the basket",
    "pick up the milk and place it in the basket",
    "pick up the orange juice and place it in the basket",
    "pick up the salad dressing and place it in the basket",
    "pick up the tomato sauce and place it in the basket"
  ],
  "libero_spatial": [
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "pick up the black bowl from table center and place it on the plate",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl next to the cookie box and place it on the plate",
    "pick up the black bowl next to the plate and place it on the plate",
    "pick up the black bowl next to the ramekin and place it on the plate",
    "pick up the black bowl on the cookie box and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
    "pick up the black bowl on the stove and place it on the plate",
    "pick up the black bowl on the wooden cabinet and place it on the plate"
  ]
}
```

- [ ] **Step 2: Add failing fixture and CLI tests**

```python
import json
from pathlib import Path


def test_all_four_libero_suites_produce_exactly_one_marker() -> None:
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "libero_task_texts.json"
    )
    suites = json.loads(fixture.read_text())
    assert {name: len(tasks) for name, tasks in suites.items()} == {
        "libero_10": 10,
        "libero_goal": 10,
        "libero_object": 10,
        "libero_spatial": 10,
    }
    marked = [
        mark_libero_target(task)
        for tasks in suites.values()
        for task in tasks
    ]
    assert len(marked) == 40
    assert all(value.count("[TGT]") == 1 for value in marked)


def test_audit_task_files_reports_each_suite(tmp_path: Path) -> None:
    from qwen3_vl_semantic_planner.audit_libero_target_text import (
        audit_task_files,
    )

    paths = []
    for suite in ("libero_10", "libero_goal"):
        path = tmp_path / suite / "meta" / "tasks.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"task_index": 0, "task": "put the bowl on the plate"})
            + "\n"
        )
        paths.append(path)
    report = audit_task_files(paths)
    assert report["total_tasks"] == 2
    assert report["total_marked"] == 2
    assert len(report["files"]) == 2
```

- [ ] **Step 3: Run the focused failures**

Run:

```bash
pytest -q tests/test_libero_target_text.py
```

Expected: the fixture test passes once the fixture exists; the CLI import fails until the audit module is created.

- [ ] **Step 4: Implement the read-only audit CLI**

```python
# qwen3_vl_semantic_planner/audit_libero_target_text.py
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
```

- [ ] **Step 5: Run the unit test and real four-suite preflight**

Run:

```bash
pytest -q tests/test_libero_target_text.py
python -m qwen3_vl_semantic_planner.audit_libero_target_text \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_10_no_noops_lerobot/meta/tasks.jsonl \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_goal_no_noops_lerobot/meta/tasks.jsonl \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_object_no_noops_lerobot/meta/tasks.jsonl \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_spatial_no_noops_lerobot/meta/tasks.jsonl
```

Expected: pytest passes; the JSON report contains `"total_tasks": 40`, `"total_marked": 40`, and `"instruction_preprocessing": "libero_tgt_v1"`.

- [ ] **Step 6: Document the mandatory preflight and planner flag**

Add to `qwen3_vl_semantic_planner/README.md`:

````markdown
### LIBERO target-aware text

New dual-camera LIBERO planner runs must pass:

```bash
--instruction-preprocessing libero_tgt_v1
```

`[TGT]` is ordinary prompt text; do not add it as a tokenizer token. Before a
long run, audit all four `meta/tasks.jsonl` files with
`python -m qwen3_vl_semantic_planner.audit_libero_target_text ...`. A new
target-aware GE-Act config must also set
`semantic_plan.instruction_preprocessing: libero_tgt_v1`; this intentionally
rejects the old unmarked planner checkpoint.
````

- [ ] **Step 7: Commit audit coverage**

```bash
git add qwen3_vl_semantic_planner/audit_libero_target_text.py \
  qwen3_vl_semantic_planner/README.md \
  tests/fixtures/libero_task_texts.json \
  tests/test_libero_target_text.py
git commit -m "test: audit LIBERO target text coverage"
```

---

