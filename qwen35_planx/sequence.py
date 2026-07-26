"""Exact teacher-forced causal layout for four TA-Tok future frames."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import torch

from qwen35_planx.config import CAMERA_NAMES, PlanGeometry
from qwen35_planx.vocabulary import (
    CAMERA_TOKENS,
    FRAME_END_TOKENS,
    FRAME_START_TOKENS,
    PLAN_END_TOKEN,
    PLAN_START_TOKEN,
    ROLE_QUERY_TOKENS,
    VisualVocabularyLayout,
)


_FIELD_PATTERNS = (
    re.compile(r"<SRC>(.*?)</SRC>", flags=re.DOTALL),
    re.compile(r"<TGT>(.*?)</TGT>", flags=re.DOTALL),
    re.compile(r"<ACT>(.*?)</ACT>", flags=re.DOTALL),
)


@dataclass(frozen=True)
class CausalPlanSequence:
    """One camera's prompt, plan tokens, and explicit hidden-state indexes."""

    camera: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    code_targets: torch.Tensor
    code_positions: torch.Tensor
    pre_positions: torch.Tensor
    post_positions: torch.Tensor
    field_positions: torch.Tensor
    field_mask: torch.Tensor
    frame_start_positions: torch.Tensor
    frame_end_positions: torch.Tensor
    plan_start_position: int
    plan_end_position: int

    @property
    def code_loss_mask(self) -> torch.Tensor:
        return self.labels.ne(-100)


def _prompt_ids(
    *,
    camera: str,
    prompt: str | Sequence[int] | torch.Tensor,
    layout: VisualVocabularyLayout,
) -> tuple[list[int], tuple[bool, bool, bool]]:
    camera_id = layout.token_id(CAMERA_TOKENS[CAMERA_NAMES.index(camera)])
    query_ids = layout.role_query_ids
    if isinstance(prompt, str):
        tokenizer = layout._tokenizer
        if tokenizer is None:
            raise ValueError("layout has no tokenizer for encoding a text prompt")
        occurrences = tuple(prompt.count(token) for token in ROLE_QUERY_TOKENS)
        if occurrences == (0, 0, 0):
            prompt = prompt + "\n" + "".join(ROLE_QUERY_TOKENS)
        elif occurrences != (1, 1, 1):
            raise ValueError("prompt must contain either zero or one of every role query")
        if prompt.count(CAMERA_TOKENS[CAMERA_NAMES.index(camera)]) == 0:
            prompt = CAMERA_TOKENS[CAMERA_NAMES.index(camera)] + "\n" + prompt
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        identifiers = [int(value) for value in encoded]
        field_mask = tuple(
            bool(match and match.group(1).strip())
            for pattern in _FIELD_PATTERNS
            for match in (pattern.search(prompt),)
        )
    else:
        values = (
            prompt.detach().cpu().reshape(-1).tolist()
            if isinstance(prompt, torch.Tensor)
            else list(prompt)
        )
        if any(type(value) is not int for value in values):
            raise TypeError("prompt token IDs must be integers")
        identifiers = [int(value) for value in values]
        query_counts = tuple(identifiers.count(token_id) for token_id in query_ids)
        if query_counts == (0, 0, 0):
            identifiers.extend(query_ids)
        elif query_counts != (1, 1, 1):
            raise ValueError(
                "prompt IDs must contain either zero or one of every role query"
            )
        if identifiers.count(camera_id) == 0:
            identifiers.insert(0, camera_id)
        field_mask = (True, True, True)

    if identifiers.count(camera_id) != 1:
        raise ValueError("prompt must contain exactly one matching camera token")
    if any(identifiers.count(token_id) != 1 for token_id in query_ids):
        raise ValueError("role queries must each be exactly one token")
    return identifiers, field_mask


def build_plan_sequence(
    *,
    camera: str,
    prompt: str | Sequence[int] | torch.Tensor,
    codes: torch.Tensor,
    layout: VisualVocabularyLayout,
    field_mask: Sequence[bool] | torch.Tensor | None = None,
) -> CausalPlanSequence:
    """Append the fixed four-frame response and expose pre/post causal indexes."""

    if camera not in CAMERA_NAMES:
        raise ValueError(f"camera must be one of {CAMERA_NAMES!r}, got {camera!r}")
    geometry = PlanGeometry()
    expected_shape = (geometry.num_keyframes, geometry.tokens_per_frame)
    if not isinstance(codes, torch.Tensor) or tuple(codes.shape) != expected_shape:
        actual = getattr(codes, "shape", None)
        raise ValueError(f"codes must have shape {expected_shape}, got {actual}")
    if codes.dtype == torch.bool or codes.dtype.is_floating_point:
        raise TypeError("codes must contain integers")
    codes = codes.detach().to(device="cpu", dtype=torch.long).contiguous()
    if codes.numel() and (
        int(codes.min()) < 0 or int(codes.max()) >= geometry.visual_vocab_size
    ):
        raise ValueError("codes must be in the released TA-Tok vocabulary range")

    identifiers, inferred_field_mask = _prompt_ids(
        camera=camera,
        prompt=prompt,
        layout=layout,
    )
    if field_mask is None:
        canonical_field_mask = inferred_field_mask
    else:
        values = (
            field_mask.detach().cpu().reshape(-1).tolist()
            if isinstance(field_mask, torch.Tensor)
            else list(field_mask)
        )
        if len(values) != 3:
            raise ValueError("field_mask must contain source, target, and action")
        canonical_field_mask = tuple(bool(value) for value in values)

    query_ids = layout.role_query_ids
    field_positions = torch.tensor(
        [identifiers.index(token_id) for token_id in query_ids],
        dtype=torch.long,
    )
    plan_start_position = len(identifiers)
    identifiers.append(layout.token_id(PLAN_START_TOKEN))

    frame_start_positions: list[int] = []
    frame_end_positions: list[int] = []
    code_positions: list[int] = []
    for frame_index in range(geometry.num_keyframes):
        frame_start_positions.append(len(identifiers))
        identifiers.append(layout.token_id(FRAME_START_TOKENS[frame_index]))
        frame_codes = codes[frame_index].tolist()
        code_positions.extend(range(len(identifiers), len(identifiers) + len(frame_codes)))
        identifiers.extend(layout.visual_start_id + code for code in frame_codes)
        frame_end_positions.append(len(identifiers))
        identifiers.append(layout.token_id(FRAME_END_TOKENS[frame_index]))
    plan_end_position = len(identifiers)
    identifiers.append(layout.token_id(PLAN_END_TOKEN))

    input_ids = torch.tensor(identifiers, dtype=torch.long)
    code_positions_tensor = torch.tensor(code_positions, dtype=torch.long)
    flattened_codes = codes.flatten()
    pre_positions = torch.empty_like(code_positions_tensor)
    pre_positions[0] = frame_start_positions[0]
    pre_positions[1:] = code_positions_tensor[:-1]
    labels = torch.full_like(input_ids, -100)
    labels[code_positions_tensor] = input_ids[code_positions_tensor]
    return CausalPlanSequence(
        camera=camera,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        code_targets=flattened_codes,
        code_positions=code_positions_tensor,
        pre_positions=pre_positions,
        post_positions=code_positions_tensor.clone(),
        field_positions=field_positions,
        field_mask=torch.tensor(canonical_field_mask, dtype=torch.bool),
        frame_start_positions=torch.tensor(frame_start_positions, dtype=torch.long),
        frame_end_positions=torch.tensor(frame_end_positions, dtype=torch.long),
        plan_start_position=plan_start_position,
        plan_end_position=plan_end_position,
    )
