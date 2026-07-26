"""Deterministic Qwen text-embedding anchors for the TA-Tok codebook."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def _decoded_token(tokenizer: Any, token_id: int) -> str:
    try:
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    return decoded if isinstance(decoded, str) else str(decoded)


def select_anchor_token_ids(
    tokenizer: Any,
    *,
    count: int = 65_536,
    seed: int = 0,
) -> tuple[int, ...]:
    """Select a stable subset of non-control base-vocabulary token IDs."""

    if count <= 0:
        raise ValueError("count must be positive")
    vocab_size = int(tokenizer.vocab_size)
    special_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
    eligible: list[int] = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        token_text = "" if token is None else str(token)
        if "<|" in token_text:
            continue
        if not _decoded_token(tokenizer, token_id).strip():
            continue
        eligible.append(token_id)

    if len(eligible) < count:
        raise ValueError(
            f"only {len(eligible)} eligible base-vocabulary tokens remain; "
            f"{count} required"
        )
    candidates = np.asarray(sorted(eligible), dtype=np.int64)
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(
        len(candidates)
    )
    selected = candidates[permutation[:count]]
    return tuple(int(token_id) for token_id in np.sort(selected))


def build_frozen_anchor_matrix(
    input_embeddings: Tensor | nn.Embedding,
    anchor_token_ids: Sequence[int],
) -> Tensor:
    """Gather anchor rows as an independent, frozen FP32 tensor."""

    weights = (
        input_embeddings.weight
        if isinstance(input_embeddings, nn.Embedding)
        else input_embeddings
    )
    if weights.ndim != 2:
        raise ValueError(
            f"input embedding matrix must be rank 2, got {weights.shape}"
        )
    token_ids = tuple(int(token_id) for token_id in anchor_token_ids)
    if not token_ids:
        raise ValueError("anchor_token_ids must not be empty")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("anchor_token_ids must be unique")
    if min(token_ids) < 0 or max(token_ids) >= weights.shape[0]:
        raise ValueError(
            f"anchor_token_ids are outside embedding row range "
            f"[0, {weights.shape[0]})"
        )

    index = torch.tensor(token_ids, dtype=torch.long, device=weights.device)
    return (
        weights.detach()
        .index_select(0, index)
        .to(dtype=torch.float32)
        .clone()
        .requires_grad_(False)
    )


def anchor_embedding_hash(anchor_matrix: Tensor) -> str:
    """Hash the exact contiguous FP32 anchor rows recorded by TA-Tok."""

    if anchor_matrix.ndim != 2:
        raise ValueError(
            f"anchor_matrix must be rank 2, got {anchor_matrix.shape}"
        )
    payload = (
        anchor_matrix.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()
