from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen3_vl_semantic_planner.libero_target_text import (
    InstructionPreprocessingError,
    LIBERO_TGT_PREPROCESSING,
    mark_libero_target,
    preprocess_libero_instructions,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "pick up the black bowl between the plate and the ramekin",
            "pick up the [TGT] black bowl between the plate and the ramekin",
        ),
        (
            "open the middle drawer of the cabinet",
            "open the [TGT] middle drawer of the cabinet",
        ),
        (
            "turn on the stove and put the moka pot on it",
            "turn on the [TGT] stove and put the moka pot on it",
        ),
        ("put the bowl on the plate", "put the [TGT] bowl on the plate"),
        (
            "pick up the alphabet soup and place it in the basket",
            "pick up the [TGT] alphabet soup and place it in the basket",
        ),
    ],
)
def test_mark_libero_target_uses_first_direct_object(
    raw: str, expected: str
) -> None:
    assert mark_libero_target(raw) == expected


def test_mark_libero_target_is_idempotent() -> None:
    marked = "open the [TGT] top drawer and put the bowl inside"
    assert mark_libero_target(marked) == marked


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "non-empty"),
        ("wait for the robot", "no target object"),
        ("pick up the [TGT] bowl and place the [TGT] bowl", "exactly one"),
    ],
)
def test_mark_libero_target_rejects_invalid_text(
    value: str, message: str
) -> None:
    with pytest.raises(InstructionPreprocessingError, match=message):
        mark_libero_target(value)


def test_mark_libero_target_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        mark_libero_target(7)  # type: ignore[arg-type]


def test_mark_libero_target_rejects_pronominal_first_object() -> None:
    with pytest.raises(InstructionPreprocessingError, match="no target object"):
        mark_libero_target("put it on the shelf and pick up the bowl")


def test_batch_preprocessing_selects_target_or_legacy_contract() -> None:
    raw = ["put the bowl on the plate"]
    assert preprocess_libero_instructions(
        raw,
        preprocessing=LIBERO_TGT_PREPROCESSING,
    ) == ["put the [TGT] bowl on the plate"]
    assert preprocess_libero_instructions(
        raw,
        preprocessing=None,
    ) == raw
    with pytest.raises(ValueError, match="unsupported instruction preprocessing"):
        preprocess_libero_instructions(raw, preprocessing="libero_tgt_v2")


def test_all_four_libero_suites_produce_exactly_one_marker() -> None:
    fixture = Path(__file__).parent / "fixtures" / "libero_task_texts.json"
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
