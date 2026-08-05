from __future__ import annotations

import re

import pytest
import torch
from torch import nn


class SequenceTokenizer:
    def __init__(self) -> None:
        self.tokens = ["<unk>", "\n"]
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        for token in payload["additional_special_tokens"]:
            self.token_to_id[token] = len(self.tokens)
            self.tokens.append(token)
        return len(payload["additional_special_tokens"])

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_to_id[token]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        specials = sorted(self.token_to_id, key=len, reverse=True)
        pattern = re.compile("|".join(re.escape(token) for token in specials))
        ids: list[int] = []
        cursor = 0
        for match in pattern.finditer(text):
            ids.extend([0] * len(text[cursor : match.start()].encode("utf-8")))
            ids.append(self.token_to_id[match.group()])
            cursor = match.end()
        ids.extend([0] * len(text[cursor:].encode("utf-8")))
        return ids


class SequenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2, 4)
        self.output = nn.Linear(4, 2, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def get_output_embeddings(self) -> nn.Linear:
        return self.output

    def resize_token_embeddings(
        self, size: int, *, mean_resizing: bool = False
    ) -> nn.Embedding:
        del mean_resizing
        old_input = self.embedding.weight.detach()
        old_output = self.output.weight.detach()
        self.embedding = nn.Embedding(size, 4)
        self.output = nn.Linear(4, size, bias=False)
        with torch.no_grad():
            self.embedding.weight[: len(old_input)].copy_(old_input)
            self.output.weight[: len(old_output)].copy_(old_output)
        return self.embedding


@pytest.fixture(scope="module")
def layout():
    from qwen35_planx.vocabulary import install_visual_vocabulary

    return install_visual_vocabulary(SequenceTokenizer(), SequenceModel())


def test_sequence_exposes_pre_and_post_positions(layout) -> None:
    from qwen35_planx.sequence import build_plan_sequence

    codes = torch.arange(4 * 729).remainder(65_536).reshape(4, 729)
    sequence = build_plan_sequence(
        camera="main",
        prompt="<ACT>pick</ACT><SRC>bowl</SRC><TGT>plate</TGT>",
        codes=codes,
        layout=layout,
    )
    assert sequence.code_targets.shape == (2916,)
    assert sequence.pre_positions.shape == (2916,)
    assert sequence.post_positions.shape == (2916,)
    assert sequence.field_positions.shape == (3,)
    assert sequence.field_mask.tolist() == [True, True, True]
    assert torch.equal(sequence.post_positions, sequence.code_positions)
    for frame_index in range(4):
        start = frame_index * 729
        end = start + 729
        assert (
            sequence.pre_positions[start]
            == sequence.frame_start_positions[frame_index]
        )
        assert torch.equal(
            sequence.pre_positions[start + 1 : end],
            sequence.code_positions[start : end - 1],
        )


def test_sequence_preserves_raster_targets_and_exact_frame_structure(layout) -> None:
    from qwen35_planx.sequence import build_plan_sequence
    from qwen35_planx.vocabulary import (
        FRAME_END_TOKENS,
        FRAME_START_TOKENS,
        PLAN_END_TOKEN,
        PLAN_START_TOKEN,
    )

    codes = torch.arange(2916).reshape(4, 729)
    sequence = build_plan_sequence(
        camera="wrist",
        prompt="<ACT>a</ACT><SRC>s</SRC><TGT>t</TGT>",
        codes=codes,
        layout=layout,
    )
    assert torch.equal(sequence.code_targets, codes.flatten())
    assert torch.equal(
        sequence.input_ids[sequence.code_positions],
        codes.flatten() + layout.visual_start_id,
    )
    assert sequence.input_ids[sequence.plan_start_position].item() == layout.token_id(
        PLAN_START_TOKEN
    )
    assert sequence.input_ids[sequence.plan_end_position].item() == layout.token_id(
        PLAN_END_TOKEN
    )
    for index in range(4):
        assert sequence.input_ids[sequence.frame_start_positions[index]].item() == (
            layout.token_id(FRAME_START_TOKENS[index])
        )
        assert sequence.input_ids[sequence.frame_end_positions[index]].item() == (
            layout.token_id(FRAME_END_TOKENS[index])
        )
        frame_codes = sequence.code_positions[index * 729 : (index + 1) * 729]
        assert torch.all(frame_codes[1:] == frame_codes[:-1] + 1)
    assert sequence.code_loss_mask.dtype == torch.bool
    assert sequence.code_loss_mask.sum().item() == 2916
    assert torch.equal(
        sequence.code_loss_mask.nonzero().flatten(), sequence.code_positions
    )


def test_role_queries_are_canonical_and_missing_roles_fail_closed(layout) -> None:
    from qwen35_planx.sequence import build_plan_sequence

    prompt = (
        "<ACT>open</ACT>\n<SRC>drawer</SRC>\n<TGT></TGT>\n"
        "Instruction: open the drawer\n"
        "<SRC_QUERY><TGT_QUERY><ACT_QUERY>\n"
        "Predict four future semantic frames."
    )
    sequence = build_plan_sequence(
        camera="main",
        prompt=prompt,
        codes=torch.zeros((4, 729), dtype=torch.long),
        layout=layout,
    )
    assert sequence.field_mask.tolist() == [True, False, True]
    assert sequence.input_ids[sequence.field_positions].tolist() == list(
        layout.role_query_ids
    )
    assert sequence.field_positions.max() < sequence.plan_start_position


def test_tokenized_prompt_without_field_evidence_fails_closed(layout) -> None:
    from qwen35_planx.sequence import build_plan_sequence

    prompt = (
        "<ACT>open</ACT><SRC>drawer</SRC><TGT></TGT>"
        "<SRC_QUERY><TGT_QUERY><ACT_QUERY>"
    )
    prompt_ids = torch.tensor(
        layout._tokenizer.encode(prompt, add_special_tokens=False)
    )
    sequence = build_plan_sequence(
        camera="main",
        prompt=prompt_ids,
        codes=torch.zeros((4, 729), dtype=torch.long),
        layout=layout,
    )
    assert sequence.field_mask.tolist() == [False, False, False]


def test_tokenized_missing_target_uses_explicit_canonical_field_mask(layout) -> None:
    from qwen35_planx.sequence import build_plan_sequence

    prompt = (
        "<ACT>open</ACT><SRC>drawer</SRC><TGT></TGT>"
        "<SRC_QUERY><TGT_QUERY><ACT_QUERY>"
    )
    prompt_ids = torch.tensor(
        layout._tokenizer.encode(prompt, add_special_tokens=False)
    )
    sequence = build_plan_sequence(
        camera="main",
        prompt=prompt_ids,
        codes=torch.zeros((4, 729), dtype=torch.long),
        layout=layout,
        field_mask=(True, False, True),
    )
    assert sequence.field_mask.tolist() == [True, False, True]


def test_camera_streams_have_independent_sequences(layout) -> None:
    from qwen35_planx.sequence import build_plan_sequence

    inputs = {
        camera: build_plan_sequence(
            camera=camera,
            prompt="<ACT>a</ACT><SRC>s</SRC><TGT>t</TGT>",
            codes=torch.zeros((4, 729), dtype=torch.long),
            layout=layout,
        )
        for camera in ("main", "wrist")
    }
    assert inputs["main"].camera == "main"
    assert inputs["wrist"].camera == "wrist"
    assert inputs["main"].input_ids.data_ptr() != inputs["wrist"].input_ids.data_ptr()
    assert not torch.equal(inputs["main"].input_ids, inputs["wrist"].input_ids)


@pytest.mark.parametrize(
    ("camera", "shape", "match"),
    [
        ("side", (4, 729), "camera"),
        ("main", (4, 728), "shape"),
    ],
)
def test_sequence_rejects_noncanonical_inputs(layout, camera, shape, match) -> None:
    from qwen35_planx.sequence import build_plan_sequence

    with pytest.raises(ValueError, match=match):
        build_plan_sequence(
            camera=camera,
            prompt="<ACT>a</ACT><SRC>s</SRC><TGT>t</TGT>",
            codes=torch.zeros(shape, dtype=torch.long),
            layout=layout,
        )
