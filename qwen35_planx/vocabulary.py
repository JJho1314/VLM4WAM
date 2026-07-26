"""Experiment-local Qwen vocabulary expansion for released TA-Tok IDs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import torch

from qwen35_planx.config import PlanGeometry
from qwen35_planx.hashing import sha256_json


PLAN_START_TOKEN = "<PLAN_START>"
PLAN_END_TOKEN = "<PLAN_END>"
FRAME_START_TOKENS = tuple(f"<FRAME_{index}>" for index in range(1, 5))
FRAME_END_TOKENS = tuple(f"</FRAME_{index}>" for index in range(1, 5))
CAMERA_TOKENS = ("<CAMERA_MAIN>", "<CAMERA_WRIST>")
ROLE_QUERY_TOKENS = ("<SRC_QUERY>", "<TGT_QUERY>", "<ACT_QUERY>")
STRUCTURE_TOKENS = (
    PLAN_START_TOKEN,
    PLAN_END_TOKEN,
    *FRAME_START_TOKENS,
    *FRAME_END_TOKENS,
    *CAMERA_TOKENS,
    *ROLE_QUERY_TOKENS,
)


def visual_token(code: int) -> str:
    """Return the exact token spelling for one released TA-Tok code."""

    size = PlanGeometry().visual_vocab_size
    if type(code) is not int or not 0 <= code < size:
        raise ValueError(f"visual code must be an integer in [0, {size})")
    return f"<|ta_{code:05d}|>"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()


def _tokenizer_hash(tokenizer: Any) -> str:
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict):
        vocabulary = dict(vocabulary)
    return sha256_json(
        sorted((str(token), int(token_id)) for token, token_id in vocabulary.items())
    )


def _is_within(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


@dataclass(frozen=True)
class VisualVocabularyLayout:
    """Immutable ID and provenance contract for one expanded experiment."""

    original_vocab_size: int
    visual_start_id: int
    visual_end_id: int
    structure_token_ids: tuple[tuple[str, int], ...]
    tokenizer_hash: str
    base_embedding_hash: str
    expanded_embedding_hash: str
    save_directory: str | None = None
    _tokenizer: Any = field(default=None, repr=False, compare=False)

    @property
    def visual_token_ids(self) -> tuple[int, ...]:
        return tuple(range(self.visual_start_id, self.visual_end_id))

    @property
    def structure_ids(self) -> tuple[int, ...]:
        return tuple(token_id for _, token_id in self.structure_token_ids)

    @property
    def role_query_ids(self) -> tuple[int, int, int]:
        identifiers = dict(self.structure_token_ids)
        return tuple(identifiers[token] for token in ROLE_QUERY_TOKENS)  # type: ignore[return-value]

    @property
    def frame_start_ids(self) -> tuple[int, int, int, int]:
        identifiers = dict(self.structure_token_ids)
        return tuple(identifiers[token] for token in FRAME_START_TOKENS)  # type: ignore[return-value]

    @property
    def frame_end_ids(self) -> tuple[int, int, int, int]:
        identifiers = dict(self.structure_token_ids)
        return tuple(identifiers[token] for token in FRAME_END_TOKENS)  # type: ignore[return-value]

    def token_id(self, token: str) -> int:
        try:
            return dict(self.structure_token_ids)[token]
        except KeyError as error:
            raise KeyError(f"unknown structural token: {token}") from error

    def code_token_id(self, code: int) -> int:
        visual_token(code)
        return self.visual_start_id + code


def install_visual_vocabulary(
    tokenizer: Any,
    model: Any,
    *,
    save_directory: Path | str | None = None,
    base_model_directory: Path | str | None = None,
) -> VisualVocabularyLayout:
    """Install visual and structural rows once, without touching base artifacts."""

    inferred_base_directories = [
        base_model_directory,
        getattr(tokenizer, "name_or_path", None),
        getattr(model, "name_or_path", None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
    ]
    if save_directory is not None:
        save_path = Path(save_directory)
        for base_value in inferred_base_directories:
            if base_value and _is_within(save_path, Path(base_value)):
                raise ValueError(
                    "save_directory must be experiment-local, "
                    "not the base model directory"
                )

    geometry = PlanGeometry()
    visual_tokens = tuple(
        f"<|ta_{code:05d}|>" for code in range(geometry.visual_vocab_size)
    )
    all_tokens = (*STRUCTURE_TOKENS, *visual_tokens)
    existing = tokenizer.get_vocab()
    collisions = tuple(token for token in all_tokens if token in existing)
    if collisions:
        raise RuntimeError(
            "visual vocabulary is already installed or collides with existing tokens"
        )

    original_vocab_size = len(tokenizer)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise TypeError("model must expose input token embeddings")
    original_input = input_embeddings.weight.detach()
    base_embedding_hash = _tensor_sha256(original_input)
    mean_row = original_input.mean(dim=0, keepdim=True)
    output_embeddings = model.get_output_embeddings()
    base_output_hash = (
        _tensor_sha256(output_embeddings.weight.detach())
        if output_embeddings is not None and hasattr(output_embeddings, "weight")
        else None
    )

    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(all_tokens)}
    )
    if added != len(all_tokens):
        raise RuntimeError(
            f"tokenizer added {added} rows, expected exactly {len(all_tokens)}"
        )
    expected_size = original_vocab_size + len(all_tokens)
    if len(tokenizer) != expected_size:
        raise RuntimeError("tokenizer size does not match installed vocabulary")

    try:
        model.resize_token_embeddings(expected_size, mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(expected_size)

    resized_input = model.get_input_embeddings().weight
    if resized_input.shape[0] != expected_size:
        raise RuntimeError("model input embeddings were not resized to tokenizer size")
    if (
        _tensor_sha256(resized_input[:original_vocab_size])
        != base_embedding_hash
    ):
        raise RuntimeError("resizing changed existing input embedding rows")
    with torch.no_grad():
        resized_input[original_vocab_size:].copy_(mean_row)

        resized_output_module = model.get_output_embeddings()
        if resized_output_module is not None:
            resized_output = resized_output_module.weight
            if resized_output.shape[0] != expected_size:
                raise RuntimeError(
                    "model output embeddings were not resized to tokenizer size"
                )
            if (
                base_output_hash is not None
                and _tensor_sha256(resized_output[:original_vocab_size])
                != base_output_hash
            ):
                raise RuntimeError("resizing changed existing output embedding rows")
            resized_output[original_vocab_size:].copy_(
                mean_row.to(device=resized_output.device, dtype=resized_output.dtype)
            )

    structure_token_ids = tuple(
        (token, int(tokenizer.convert_tokens_to_ids(token)))
        for token in STRUCTURE_TOKENS
    )
    visual_start_id = int(tokenizer.convert_tokens_to_ids(visual_tokens[0]))
    visual_end_id = int(tokenizer.convert_tokens_to_ids(visual_tokens[-1])) + 1
    if tuple(token_id for _, token_id in structure_token_ids) != tuple(
        range(original_vocab_size, original_vocab_size + len(STRUCTURE_TOKENS))
    ):
        raise RuntimeError("structural token IDs are not unique and contiguous")
    if visual_end_id - visual_start_id != geometry.visual_vocab_size:
        raise RuntimeError("visual token IDs are not contiguous")
    for token, expected_id in (
        (visual_tokens[0], visual_start_id),
        (visual_tokens[-1], visual_end_id - 1),
        *structure_token_ids,
    ):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [expected_id]:
            raise RuntimeError(f"installed token is not single-token: {token}")

    if save_directory is not None:
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(destination)
        model.save_pretrained(destination)

    return VisualVocabularyLayout(
        original_vocab_size=original_vocab_size,
        visual_start_id=visual_start_id,
        visual_end_id=visual_end_id,
        structure_token_ids=structure_token_ids,
        tokenizer_hash=_tokenizer_hash(tokenizer),
        base_embedding_hash=base_embedding_hash,
        expanded_embedding_hash=_tensor_sha256(resized_input),
        save_directory=(
            str(Path(save_directory).resolve()) if save_directory is not None else None
        ),
        _tokenizer=tokenizer,
    )
