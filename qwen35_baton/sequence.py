"""Fixed Qwen text layout for continuous Baton plan queries."""

from __future__ import annotations

import torch


PLAN_START = "<PLAN_START>"
FRAME_TOKENS = tuple(f"<FRAME_{index}>" for index in range(4))
PLAN_PAD = "<PLAN_PAD>"
PLAN_END = "<PLAN_END>"
ADDED_TOKENS = (PLAN_START, *FRAME_TOKENS, PLAN_PAD, PLAN_END)


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
