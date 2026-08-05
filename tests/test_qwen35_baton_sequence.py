from __future__ import annotations

import pytest


WORLD_ARENA_PREFIX = (
    "In a fixed robotic workspace, generate a rigid, physically consistent "
    "embodied robotic arm. The arm maintains high stability with no deformation "
    "and enters the frame to "
)


def test_baton_conversation_places_blueprint_in_assistant_with_time() -> None:
    from qwen35_baton.sequence import (
        PLAN_PAD,
        build_baton_conversation,
    )

    conversation = build_baton_conversation(
        instruction="Pick up the red cube",
        source_indices=(12, 39, 66, 93, 120),
    )

    assert [message["role"] for message in conversation] == [
        "system",
        "user",
        "assistant",
    ]
    assert conversation[1]["content"][0]["type"] == "image"
    user_text = conversation[1]["content"][1]["text"]
    assert "Instruction: Pick up the red cube" in user_text
    assert "Current frame: 12/120, normalized time 0.100000." in user_text
    assert "Target frames: 39/120, 66/120, 93/120, 120/120." in user_text
    assistant_text = conversation[2]["content"]
    assert assistant_text.count(PLAN_PAD) == 4 * 256
    assert "Instruction:" not in assistant_text
    assert assistant_text.startswith("<PLAN_START>\n")
    assert assistant_text.endswith("\n<PLAN_END>")


def test_legacy_plan_text_is_byte_for_byte_unchanged() -> None:
    from qwen35_baton.sequence import PLAN_PAD, build_plan_text

    expected_blocks = [
        f"<FRAME_{index}> " + " ".join([PLAN_PAD] * 256)
        for index in range(4)
    ]
    expected = (
        "Instruction: pick up the red cube\n"
        "<PLAN_START>\n"
        + "\n".join(expected_blocks)
        + "\n<PLAN_END>"
    )

    assert build_plan_text("pick up the red cube") == expected


@pytest.mark.parametrize(
    "source_indices",
    [
        (12, 39, 66, 93),
        (12, 39, 39, 93, 120),
        (-1, 39, 66, 93, 120),
        (12, 39, 66, 93, 121),
        (12, 39, 66, 93, 120.0),
    ],
)
def test_baton_conversation_rejects_invalid_source_indices(
    source_indices: object,
) -> None:
    from qwen35_baton.sequence import build_baton_conversation

    with pytest.raises(ValueError, match="source_indices"):
        build_baton_conversation(
            instruction="pick up the red cube",
            source_indices=source_indices,
        )


def test_worldarena_instruction_rendering_strips_only_exact_prefix() -> None:
    from qwen35_baton.sequence import (
        STRIP_WORLD_ARENA_INSTRUCTION_KIND,
        VERBATIM_INSTRUCTION_KIND,
        render_instruction,
    )

    original = WORLD_ARENA_PREFIX + "Pick up the red cube"
    assert (
        render_instruction(original, STRIP_WORLD_ARENA_INSTRUCTION_KIND)
        == "Pick up the red cube"
    )
    assert render_instruction(original, VERBATIM_INSTRUCTION_KIND) == original
    near_match = WORLD_ARENA_PREFIX.replace("fixed", "static") + "Pick it up"
    assert render_instruction(near_match, STRIP_WORLD_ARENA_INSTRUCTION_KIND) == near_match


def test_worldarena_instruction_rendering_rejects_blank_remainder() -> None:
    from qwen35_baton.sequence import (
        STRIP_WORLD_ARENA_INSTRUCTION_KIND,
        render_instruction,
    )

    with pytest.raises(ValueError, match="nonblank task clause"):
        render_instruction(
            WORLD_ARENA_PREFIX + "   ",
            STRIP_WORLD_ARENA_INSTRUCTION_KIND,
        )
