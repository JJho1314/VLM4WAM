from __future__ import annotations

import pytest
import torch


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
