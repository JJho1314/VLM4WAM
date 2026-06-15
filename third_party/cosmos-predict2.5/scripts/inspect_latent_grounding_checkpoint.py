#!/usr/bin/env python3
"""Inspect latent-grounding gates in a Cosmos DCP checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint iter directory or model subdir.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ckpt_dir = args.checkpoint
    model_dir = ckpt_dir / "model" if (ckpt_dir / "model").is_dir() else ckpt_dir
    metadata = dcp.FileSystemReader(str(model_dir)).read_metadata()
    state_keys = sorted(metadata.state_dict_metadata.keys())
    gate_keys = [key for key in state_keys if "target_latent_gate" in key]
    related_keys = [
        key
        for key in state_keys
        if "target_latent" in key or "target_latent_grounding" in key
    ]

    state = {key: torch.empty(tuple(metadata.state_dict_metadata[key].size), dtype=torch.float32) for key in gate_keys}
    values = {}
    if state:
        dcp.load_state_dict(state, storage_reader=dcp.FileSystemReader(str(model_dir)), no_dist=True)
        for key, tensor in state.items():
            tensor = tensor.detach().cpu().float()
            values[key] = {
                "raw": tensor.flatten().tolist(),
                "tanh": torch.tanh(tensor).flatten().tolist(),
                "abs_mean": float(tensor.abs().mean().item()),
                "abs_tanh_mean": float(torch.tanh(tensor).abs().mean().item()),
            }

    summary = {
        "checkpoint": str(args.checkpoint),
        "model_dir": str(model_dir),
        "num_state_keys": len(state_keys),
        "gate_keys": gate_keys,
        "related_keys": related_keys,
        "gate_values": values,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
