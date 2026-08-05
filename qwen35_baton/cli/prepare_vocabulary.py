"""Create an experiment-local Qwen3.5 artifact with the seven Baton tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from qwen35_baton.sequence import ADDED_TOKENS


def _is_within(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def install_baton_vocabulary(
    tokenizer: Any,
    model: Any,
    processor: Any,
    *,
    destination: str | Path,
    base_model_directory: str | Path,
) -> tuple[int, ...]:
    """Add exactly the Baton structure rows and persist a self-contained copy."""

    destination = Path(destination)
    base_model_directory = Path(base_model_directory)
    if _is_within(destination, base_model_directory):
        raise ValueError(
            "destination must be outside the base model directory"
        )
    collisions = tuple(
        token for token in ADDED_TOKENS if token in tokenizer.get_vocab()
    )
    if collisions:
        raise RuntimeError(
            "Baton vocabulary collides with existing tokens: "
            + ", ".join(collisions)
        )

    original_size = len(tokenizer)
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if input_embedding is None or not hasattr(input_embedding, "weight"):
        raise TypeError("model must expose input token embeddings")
    original_input = input_embedding.weight.detach().clone()
    original_output = (
        None
        if output_embedding is None or not hasattr(output_embedding, "weight")
        else output_embedding.weight.detach().clone()
    )

    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(ADDED_TOKENS)}
    )
    if added != len(ADDED_TOKENS) or len(tokenizer) != original_size + added:
        raise RuntimeError("tokenizer did not add exactly seven Baton tokens")
    expected_size = original_size + len(ADDED_TOKENS)
    model_vocab_size = int(original_input.shape[0])
    if expected_size > model_vocab_size:
        try:
            model.resize_token_embeddings(expected_size, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(expected_size)

    resized_input = model.get_input_embeddings().weight
    expected_embedding_size = max(model_vocab_size, expected_size)
    if resized_input.shape[0] != expected_embedding_size:
        raise RuntimeError("resized input embedding has the wrong vocabulary size")
    if not torch.equal(resized_input[:model_vocab_size], original_input):
        raise RuntimeError("vocabulary resize changed existing input rows")
    resized_output_module = model.get_output_embeddings()
    resized_output = (
        None
        if resized_output_module is None
        else resized_output_module.weight
    )
    if resized_output is not None:
        if resized_output.shape[0] != expected_embedding_size:
            raise RuntimeError(
                "resized output embedding has the wrong vocabulary size"
            )
        if original_output is not None and not torch.equal(
            resized_output[:model_vocab_size], original_output
        ):
            raise RuntimeError("vocabulary resize changed existing output rows")
    input_mean = original_input.mean(dim=0, keepdim=True)
    with torch.no_grad():
        resized_input[original_size:expected_size].copy_(
            input_mean.to(device=resized_input.device, dtype=resized_input.dtype)
        )
        if resized_output is not None:
            if resized_output.data_ptr() != resized_input.data_ptr():
                output_mean = (
                    input_mean
                    if original_output is None
                    else original_output.mean(dim=0, keepdim=True)
                )
                resized_output[original_size:expected_size].copy_(
                    output_mean.to(
                        device=resized_output.device,
                        dtype=resized_output.dtype,
                    )
                )

    token_ids = tuple(
        int(tokenizer.convert_tokens_to_ids(token)) for token in ADDED_TOKENS
    )
    if token_ids != tuple(range(original_size, expected_size)):
        raise RuntimeError("Baton token IDs are not unique and contiguous")
    for token, token_id in zip(ADDED_TOKENS, token_ids, strict=True):
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise RuntimeError(f"Baton token is not single-token: {token}")

    destination.mkdir(parents=True, exist_ok=False)
    processor.tokenizer = tokenizer
    tokenizer.save_pretrained(destination)
    processor.save_pretrained(destination)
    model.save_pretrained(
        destination,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    return token_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(
        args.base_model, local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    token_ids = install_baton_vocabulary(
        tokenizer,
        model,
        processor,
        destination=args.output,
        base_model_directory=args.base_model,
    )
    print(
        json.dumps(
            {
                "base_model": str(args.base_model.resolve()),
                "output": str(args.output.resolve()),
                "added_tokens": dict(zip(ADDED_TOKENS, token_ids, strict=True)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
