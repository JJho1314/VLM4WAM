from __future__ import annotations

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
