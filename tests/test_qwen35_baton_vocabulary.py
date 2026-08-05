from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from qwen35_baton.sequence import ADDED_TOKENS


class FakeTokenizer:
    def __init__(self, size: int = 8) -> None:
        self.tokens = [f"base_{index}" for index in range(size)]
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.saved_to: Path | None = None

    def __len__(self) -> int:
        return len(self.tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.tokens)
                self.tokens.append(token)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_to_id[token]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.token_to_id[text]]

    def save_pretrained(self, path: str | Path) -> None:
        self.saved_to = Path(path)


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = None
        self.saved_to: Path | None = None

    def save_pretrained(self, path: str | Path) -> None:
        self.saved_to = Path(path)


class FakeQwen(nn.Module):
    def __init__(self, size: int = 8, width: int = 4) -> None:
        super().__init__()
        weight = torch.arange(size * width, dtype=torch.float32).reshape(
            size, width
        )
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)
        self.output = nn.Linear(width, size, bias=False)
        self.output.weight = self.embedding.weight
        self.saved_to: Path | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def get_output_embeddings(self) -> nn.Linear:
        return self.output

    def resize_token_embeddings(
        self, size: int, *, mean_resizing: bool = False
    ) -> nn.Embedding:
        del mean_resizing
        old = self.embedding.weight.detach()
        replacement = nn.Embedding(size, old.shape[1], dtype=old.dtype)
        with torch.no_grad():
            replacement.weight[: len(old)].copy_(old)
        self.embedding = replacement
        output = nn.Linear(old.shape[1], size, bias=False, dtype=old.dtype)
        output.weight = self.embedding.weight
        self.output = output
        return replacement

    def save_pretrained(self, path: str | Path, **_: object) -> None:
        self.saved_to = Path(path)


def test_baton_vocabulary_adds_only_seven_mean_initialized_rows(tmp_path: Path):
    from qwen35_baton.cli.prepare_vocabulary import install_baton_vocabulary

    tokenizer = FakeTokenizer()
    processor = FakeProcessor()
    model = FakeQwen()
    old = model.get_input_embeddings().weight.detach().clone()
    destination = tmp_path / "baton"

    token_ids = install_baton_vocabulary(
        tokenizer,
        model,
        processor,
        destination=destination,
        base_model_directory=tmp_path / "base",
    )

    assert token_ids == tuple(range(len(old), len(old) + len(ADDED_TOKENS)))
    assert len(tokenizer) == len(old) + 7
    assert torch.equal(model.get_input_embeddings().weight[: len(old)], old)
    torch.testing.assert_close(
        model.get_input_embeddings().weight[len(old) :],
        old.mean(0, keepdim=True).expand(7, -1),
    )
    assert model.get_input_embeddings().weight.data_ptr() == (
        model.get_output_embeddings().weight.data_ptr()
    )
    assert processor.tokenizer is tokenizer
    assert tokenizer.saved_to == destination
    assert processor.saved_to == destination
    assert model.saved_to == destination
    for token, token_id in zip(ADDED_TOKENS, token_ids, strict=True):
        assert tokenizer.encode(token, add_special_tokens=False) == [token_id]


def test_baton_vocabulary_refuses_base_overwrite_and_token_collisions(
    tmp_path: Path,
):
    from qwen35_baton.cli.prepare_vocabulary import install_baton_vocabulary

    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(ValueError, match="base model directory"):
        install_baton_vocabulary(
            FakeTokenizer(),
            FakeQwen(),
            FakeProcessor(),
            destination=base,
            base_model_directory=base,
        )

    tokenizer = FakeTokenizer()
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [ADDED_TOKENS[0]]}
    )
    with pytest.raises(RuntimeError, match="collides"):
        install_baton_vocabulary(
            tokenizer,
            FakeQwen(),
            FakeProcessor(),
            destination=tmp_path / "baton",
            base_model_directory=base,
        )


def test_baton_vocabulary_reuses_padded_embedding_without_shrinking(
    tmp_path: Path,
):
    from qwen35_baton.cli.prepare_vocabulary import install_baton_vocabulary

    tokenizer = FakeTokenizer(size=8)
    model = FakeQwen(size=16)
    original = model.get_input_embeddings().weight.detach().clone()

    token_ids = install_baton_vocabulary(
        tokenizer,
        model,
        FakeProcessor(),
        destination=tmp_path / "baton",
        base_model_directory=tmp_path / "base",
    )

    assert token_ids == tuple(range(8, 15))
    assert model.get_input_embeddings().weight.shape == (16, 4)
    assert torch.equal(model.get_input_embeddings().weight[:8], original[:8])
    assert torch.equal(model.get_input_embeddings().weight[15], original[15])
    torch.testing.assert_close(
        model.get_input_embeddings().weight[8:15],
        original.mean(0, keepdim=True).expand(7, -1),
    )
