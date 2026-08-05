"""Fixed Qwen text layout for continuous Baton plan queries."""

from __future__ import annotations

import torch


PLAN_START = "<PLAN_START>"
FRAME_TOKENS = tuple(f"<FRAME_{index}>" for index in range(4))
PLAN_PAD = "<PLAN_PAD>"
PLAN_END = "<PLAN_END>"
ADDED_TOKENS = (PLAN_START, *FRAME_TOKENS, PLAN_PAD, PLAN_END)

LEGACY_TEMPLATE_KIND = "legacy_user_plan_v1"
BATON_TEMPLATE_KIND = "baton_assistant_time_v2"
VERBATIM_INSTRUCTION_KIND = "verbatim_v1"
STRIP_WORLD_ARENA_INSTRUCTION_KIND = "strip_worldarena_boilerplate_v1"

BATON_SYSTEM_TEXT = (
    "You are a helpful assistant that predicts spatially grounded future visual "
    "semantic blueprints for embodied robot videos."
)
WORLD_ARENA_BOILERPLATE_PREFIX = (
    "In a fixed robotic workspace, generate a rigid, physically consistent "
    "embodied robotic arm. The arm maintains high stability with no deformation "
    "and enters the frame to "
)


def build_plan_text(instruction: str) -> str:
    """Append the fixed four-frame, 1,024-query plan template."""

    if not isinstance(instruction, str) or not instruction:
        raise ValueError("instruction must be a nonempty string")
    blocks = [
        f"{FRAME_TOKENS[index]} " + " ".join([PLAN_PAD] * 256) for index in range(4)
    ]
    return (
        f"Instruction: {instruction}\n"
        f"{PLAN_START}\n" + "\n".join(blocks) + f"\n{PLAN_END}"
    )


def build_plan_scaffold() -> str:
    """Return the fixed assistant-side four-frame blueprint scaffold."""

    blocks = [
        f"{FRAME_TOKENS[index]} " + " ".join([PLAN_PAD] * 256)
        for index in range(4)
    ]
    return f"{PLAN_START}\n" + "\n".join(blocks) + f"\n{PLAN_END}"


def validate_source_indices(
    source_indices: object,
) -> tuple[int, int, int, int, int]:
    """Validate current plus four strictly increasing canonical frame indices."""

    if (
        not isinstance(source_indices, tuple)
        or len(source_indices) != 5
        or any(type(value) is not int for value in source_indices)
        or not 0 <= source_indices[0]
        or not all(left < right for left, right in zip(source_indices, source_indices[1:]))
        or source_indices[-1] > 120
    ):
        raise ValueError(
            "source_indices must be five strictly increasing integers in [0,120]"
        )
    return source_indices


def render_instruction(instruction: str, rendering_kind: str) -> str:
    """Render one auditable instruction according to a versioned policy."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a nonempty string")
    if rendering_kind == VERBATIM_INSTRUCTION_KIND:
        return instruction
    if rendering_kind != STRIP_WORLD_ARENA_INSTRUCTION_KIND:
        raise ValueError(f"unsupported instruction rendering kind: {rendering_kind!r}")
    if not instruction.startswith(WORLD_ARENA_BOILERPLATE_PREFIX):
        return instruction
    task_clause = instruction[len(WORLD_ARENA_BOILERPLATE_PREFIX) :].strip()
    if not task_clause:
        raise ValueError("WorldArena instruction must retain a nonblank task clause")
    return task_clause


def build_baton_user_text(
    instruction: str,
    source_indices: tuple[int, int, int, int, int],
) -> str:
    """Describe the requested future horizon with explicit canonical time."""

    current, *targets = validate_source_indices(source_indices)
    rendered_targets = ", ".join(f"{target}/120" for target in targets)
    return (
        "Predict four future semantic keyframes for this observation.\n"
        f"Instruction: {instruction}\n"
        f"Current frame: {current}/120, normalized time {current / 120:.6f}.\n"
        f"Target frames: {rendered_targets}."
    )


def build_baton_conversation(
    instruction: str,
    source_indices: tuple[int, int, int, int, int],
) -> list[dict[str, object]]:
    """Build the complete Baton-style system/user/assistant conversation."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a nonempty string")
    indices = validate_source_indices(source_indices)
    return [
        {"role": "system", "content": BATON_SYSTEM_TEXT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": build_baton_user_text(instruction, indices),
                },
            ],
        },
        {"role": "assistant", "content": build_plan_scaffold()},
    ]


def input_template_contract(template_kind: str) -> object:
    """Return the canonical hash payload for a versioned input template."""

    if template_kind == LEGACY_TEMPLATE_KIND:
        return build_plan_text("{instruction}")
    if template_kind == BATON_TEMPLATE_KIND:
        return {
            "kind": BATON_TEMPLATE_KIND,
            "system": BATON_SYSTEM_TEXT,
            "user": (
                "Predict four future semantic keyframes for this observation.\n"
                "Instruction: {instruction}\n"
                "Current frame: {current}/120, normalized time {normalized_time:.6f}.\n"
                "Target frames: {f0}/120, {f1}/120, {f2}/120, {f3}/120."
            ),
            "assistant": build_plan_scaffold(),
            "add_generation_prompt": False,
        }
    raise ValueError(f"unsupported input template kind: {template_kind!r}")


def find_plan_positions(
    input_ids: torch.Tensor, plan_pad_token_id: int
) -> torch.Tensor:
    """Return the 1,024 plan-pad positions for every Qwen row, or fail closed."""

    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.dtype == torch.bool
        or input_ids.dtype.is_floating_point
    ):
        raise ValueError("input_ids must be a rank-2 integer tensor")
    if type(plan_pad_token_id) is not int or plan_pad_token_id < 0:
        raise ValueError("plan_pad_token_id must be a non-negative integer")

    positions = tuple(
        torch.nonzero(row.eq(plan_pad_token_id), as_tuple=False).flatten()
        for row in input_ids
    )
    if not positions or any(position.numel() != 1024 for position in positions):
        raise ValueError("each input row must contain exactly 1024 <PLAN_PAD> tokens")
    return torch.stack(positions)
