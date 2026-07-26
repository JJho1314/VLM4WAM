from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab_size = 20
        self.all_special_ids = [0, 1]
        self._tokens = {
            0: "<bos>",
            1: "<eos>",
            2: "<|reserved|>",
            3: "blank",
            **{index: f"token-{index}" for index in range(4, 20)},
            20: "<|added-control|>",
        }
        self._decoded = {
            index: token for index, token in self._tokens.items()
        }
        self._decoded[3] = "   "

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._tokens[token_id]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self._decoded[token_id] for token_id in token_ids)


def test_anchor_selection_is_deterministic_and_excludes_controls() -> None:
    from qwen35_planx.anchors import select_anchor_token_ids

    tokenizer = FakeTokenizer()
    first = select_anchor_token_ids(tokenizer, count=8, seed=17)
    second = select_anchor_token_ids(tokenizer, count=8, seed=17)

    assert first == second
    assert first == tuple(sorted(first))
    assert not set(first).intersection(tokenizer.all_special_ids)
    assert 2 not in first
    assert 3 not in first
    assert all(
        "<|" not in tokenizer.convert_ids_to_tokens(token_id)
        for token_id in first
    )


def test_anchor_selection_fails_when_too_few_base_tokens_survive() -> None:
    from qwen35_planx.anchors import select_anchor_token_ids

    with pytest.raises(ValueError, match="eligible"):
        select_anchor_token_ids(FakeTokenizer(), count=18, seed=0)


def test_anchor_matrix_is_fp32_frozen_and_hash_is_stable() -> None:
    from qwen35_planx.anchors import (
        anchor_embedding_hash,
        build_frozen_anchor_matrix,
    )

    embedding = torch.arange(80, dtype=torch.float16).reshape(20, 4)
    matrix = build_frozen_anchor_matrix(embedding, [2, 5, 7])

    assert matrix.shape == (3, 4)
    assert matrix.dtype == torch.float32
    assert matrix.requires_grad is False
    assert torch.equal(matrix, embedding[[2, 5, 7]].float())
    assert anchor_embedding_hash(matrix) == anchor_embedding_hash(matrix.clone())


def test_anchor_matrix_rejects_duplicate_and_out_of_range_ids() -> None:
    from qwen35_planx.anchors import build_frozen_anchor_matrix

    embedding = torch.arange(20).reshape(5, 4)
    with pytest.raises(ValueError, match="unique"):
        build_frozen_anchor_matrix(embedding, [1, 1])
    with pytest.raises(ValueError, match="range"):
        build_frozen_anchor_matrix(embedding, [5])


class FakeSiglip(nn.Module):
    def __init__(self, feature_dim: int = 8, tokens: int = 256) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.feature_dim = feature_dim
        self.tokens = tokens

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert output_hidden_states and return_dict
        batch = pixel_values.shape[0]
        signal = pixel_values.mean(dim=(1, 2, 3), keepdim=True).reshape(
            batch, 1, 1
        )
        positions = torch.linspace(
            0.1,
            1.0,
            self.tokens,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).reshape(1, self.tokens, 1)
        channels = torch.linspace(
            0.2,
            1.2,
            self.feature_dim,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).reshape(1, 1, self.feature_dim)
        penultimate = (signal + positions + channels) * self.scale
        final = torch.full_like(penultimate, 99.0)
        return SimpleNamespace(
            hidden_states=(torch.zeros_like(penultimate), penultimate, final)
        )


def _make_ta_tok(*, tokens: int = 256):
    from qwen35_planx.ta_tok import TextAlignedTokenizer

    torch.manual_seed(0)
    return TextAlignedTokenizer.from_modules(
        student=FakeSiglip(tokens=tokens),
        teacher=FakeSiglip(tokens=tokens),
        frozen_anchors=torch.randn(32, 12),
        feature_dim=8,
        qwen_dim=12,
        decoder_depth=3,
        decoder_num_heads=2,
        codebook_chunk_size=7,
    )


def test_chunked_nearest_codes_match_dense_cosine_and_tie_break_low() -> None:
    from qwen35_planx.ta_tok import nearest_code_indices

    torch.manual_seed(3)
    queries = torch.randn(2, 5, 7)
    codebook = torch.randn(19, 7)
    expected = (
        torch.nn.functional.normalize(queries, dim=-1)
        @ torch.nn.functional.normalize(codebook, dim=-1).T
    ).argmax(dim=-1)

    actual = nearest_code_indices(queries, codebook, codebook_chunk_size=4)
    assert torch.equal(actual, expected)

    tied = nearest_code_indices(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
        codebook_chunk_size=1,
    )
    assert tied.item() == 0


def test_ta_tok_shapes_losses_and_stop_gradient() -> None:
    model = _make_ta_tok()
    output = model(torch.rand(2, 3, 224, 224))

    assert output.codes.shape == (2, 256)
    assert output.quantized.shape == (2, 256, 12)
    assert output.reconstruction.shape == (2, 256, 8)
    assert set(output.losses) == {"reconstruction", "commitment", "codebook"}
    assert all(loss.ndim == 0 for loss in output.losses.values())
    assert len(model.decoder_blocks.layers) == 3

    output.loss.backward()
    assert model.student_projection.weight.grad is not None
    assert model.codebook_projection.weight.grad is not None
    assert model.frozen_anchors.grad is None
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    assert model.student.scale.grad is not None


def test_ta_tok_uses_penultimate_hidden_state_and_keeps_teacher_eval() -> None:
    model = _make_ta_tok()
    images = torch.full((1, 3, 256, 256), 0.75)

    features = model.extract_student_features(images)
    assert features.shape == (1, 256, 8)
    assert not torch.all(features == 99.0)

    model.train()
    assert model.student.training
    assert not model.teacher.training
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())


def test_ta_tok_rejects_non_256_token_siglip_output() -> None:
    model = _make_ta_tok(tokens=255)
    with pytest.raises(ValueError, match="256"):
        model(torch.rand(1, 3, 256, 256))


def test_ta_tok_metrics_and_encode_decode_round_trip_shapes() -> None:
    from qwen35_planx.ta_tok import codebook_usage_metrics

    metrics = codebook_usage_metrics(
        torch.tensor([[0, 0, 1, 1], [1, 2, 2, 2]]),
        vocabulary_size=4,
    )
    assert metrics["code_usage"].item() == pytest.approx(0.75)
    assert metrics["dead_code_ratio"].item() == pytest.approx(0.25)
    assert 1.0 < metrics["perplexity"].item() <= 3.0

    model = _make_ta_tok()
    images = torch.rand(2, 3, 256, 256)
    codes = model.encode_codes(images)
    reconstruction = model.decode_codes(codes)
    assert codes.shape == (2, 256)
    assert reconstruction.shape == (2, 256, 8)
