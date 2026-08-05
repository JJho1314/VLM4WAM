from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn


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


class FakeQwen(nn.Module):
    def __init__(self, size: int = 8, width: int = 2048) -> None:
        super().__init__()
        weight = torch.arange(size * width, dtype=torch.float16).reshape(size, width)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)
        self.output = nn.Linear(width, size, bias=False)
        self.output.weight = self.embedding.weight
        self.resize_calls = 0
        self.saved_to: Path | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def get_output_embeddings(self) -> nn.Linear:
        return self.output

    def resize_token_embeddings(
        self, size: int, *, mean_resizing: bool = False
    ) -> nn.Embedding:
        del mean_resizing
        self.resize_calls += 1
        old = self.embedding.weight.detach()
        replacement = nn.Embedding(size, old.shape[1], dtype=old.dtype)
        with torch.no_grad():
            replacement.weight[: len(old)].copy_(old)
        self.embedding = replacement
        output = nn.Linear(old.shape[1], size, bias=False, dtype=old.dtype)
        output.weight = self.embedding.weight
        self.output = output
        return replacement

    def save_pretrained(self, path: str | Path) -> None:
        self.saved_to = Path(path)


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def fake_qwen() -> FakeQwen:
    return FakeQwen()


def test_visual_rows_use_mean_qwen_initialization(
    fake_qwen: FakeQwen, fake_tokenizer: FakeTokenizer
) -> None:
    from qwen35_planx.vocabulary import install_visual_vocabulary

    old = fake_qwen.get_input_embeddings().weight.detach().clone()
    layout = install_visual_vocabulary(fake_tokenizer, fake_qwen)
    new_rows = fake_qwen.get_input_embeddings().weight[
        layout.visual_start_id : layout.visual_end_id
    ]
    torch.testing.assert_close(
        new_rows,
        old.mean(0, keepdim=True).expand_as(new_rows),
    )
    assert new_rows.shape == (65_536, 2048)
    assert not hasattr(layout, "codebook_projection")


def test_installation_preserves_base_rows_and_builds_exact_single_token_layout(
    fake_qwen: FakeQwen, fake_tokenizer: FakeTokenizer
) -> None:
    from qwen35_planx.vocabulary import (
        ROLE_QUERY_TOKENS,
        STRUCTURE_TOKENS,
        install_visual_vocabulary,
    )

    old_input = fake_qwen.get_input_embeddings().weight.detach().clone()
    old_output = fake_qwen.get_output_embeddings().weight.detach().clone()
    layout = install_visual_vocabulary(fake_tokenizer, fake_qwen)

    assert fake_qwen.resize_calls == 1
    assert layout.visual_token_ids == tuple(
        range(layout.visual_start_id, layout.visual_end_id)
    )
    assert layout.visual_end_id - layout.visual_start_id == 65_536
    assert len(set(layout.structure_ids)) == len(STRUCTURE_TOKENS)
    assert layout.role_query_ids == tuple(
        fake_tokenizer.convert_tokens_to_ids(token) for token in ROLE_QUERY_TOKENS
    )
    for token in (*STRUCTURE_TOKENS, "<|ta_00000|>", "<|ta_65535|>"):
        assert fake_tokenizer.encode(token, add_special_tokens=False) == [
            fake_tokenizer.convert_tokens_to_ids(token)
        ]
    assert torch.equal(
        fake_qwen.get_input_embeddings().weight[: len(old_input)], old_input
    )
    assert torch.equal(
        fake_qwen.get_output_embeddings().weight[: len(old_output)], old_output
    )
    assert fake_qwen.get_input_embeddings().weight.data_ptr() == (
        fake_qwen.get_output_embeddings().weight.data_ptr()
    )
    assert len(layout.tokenizer_hash) == 64
    assert len(layout.base_embedding_hash) == 64
    assert len(layout.expanded_embedding_hash) == 64


def test_installation_is_one_shot_and_saves_only_to_experiment_directory(
    tmp_path: Path, fake_qwen: FakeQwen, fake_tokenizer: FakeTokenizer
) -> None:
    from qwen35_planx.vocabulary import install_visual_vocabulary

    base = tmp_path / "base"
    experiment = tmp_path / "experiment"
    base.mkdir()
    install_visual_vocabulary(
        fake_tokenizer,
        fake_qwen,
        save_directory=experiment,
        base_model_directory=base,
    )
    assert fake_tokenizer.saved_to == experiment
    assert fake_qwen.saved_to == experiment
    with pytest.raises(RuntimeError, match="already"):
        install_visual_vocabulary(fake_tokenizer, fake_qwen)

    other_tokenizer = FakeTokenizer()
    other_qwen = FakeQwen(width=4)
    with pytest.raises(ValueError, match="base model directory"):
        install_visual_vocabulary(
            other_tokenizer,
            other_qwen,
            save_directory=base,
            base_model_directory=base,
        )
    assert other_qwen.resize_calls == 0


def test_installer_infers_and_protects_the_base_model_directory(
    tmp_path: Path,
) -> None:
    from qwen35_planx.vocabulary import install_visual_vocabulary

    tokenizer = FakeTokenizer()
    tokenizer.name_or_path = str(tmp_path / "base")
    model = FakeQwen(width=4)
    with pytest.raises(ValueError, match="base model directory"):
        install_visual_vocabulary(
            tokenizer,
            model,
            save_directory=tmp_path / "base",
        )
    assert model.resize_calls == 0
