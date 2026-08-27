#!/usr/bin/env python3
"""Build discrete semantic-token labels from continuous Qwen3-VL features.

Plan-X uses TA-Tok discrete tokens as planner targets.  In this project we do
not have TA-Tok in the Cosmos stack, so this script builds an experiment-local
vector-quantized codebook over pooled Qwen3-VL image-token features and emits
future semantic code IDs for Stage-A autoregressive CE training.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-train-tokens", type=int, default=262144)
    parser.add_argument("--kmeans-iters", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_label_paths(label_dir: Path, max_samples: int) -> list[Path]:
    manifest = label_dir / "manifest.jsonl"
    if manifest.exists():
        paths = []
        for line in manifest.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                p = Path(rec["path"])
                if p.exists():
                    paths.append(p)
    else:
        paths = sorted(label_dir.glob("*.pt"))
    if max_samples > 0:
        paths = paths[:max_samples]
    if not paths:
        raise RuntimeError(f"No continuous labels found under {label_dir}")
    return paths


def load_plan(path: Path) -> tuple[dict[str, Any], torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    plan = payload["semantic_plan"].float().reshape(-1, payload["semantic_plan"].shape[-1])
    return payload, plan


def maybe_normalize(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    return F.normalize(x, dim=-1) if enabled else x


def collect_training_tokens(paths: list[Path], max_tokens: int, normalize: bool, seed: int) -> torch.Tensor:
    random.seed(seed)
    chunks = []
    total = 0
    shuffled = list(paths)
    random.shuffle(shuffled)
    for path in tqdm(shuffled, desc="collect tokens"):
        _, plan = load_plan(path)
        plan = maybe_normalize(plan, normalize)
        chunks.append(plan)
        total += plan.shape[0]
        if total >= max_tokens:
            break
    x = torch.cat(chunks, dim=0)
    if x.shape[0] > max_tokens:
        gen = torch.Generator().manual_seed(seed)
        keep = torch.randperm(x.shape[0], generator=gen)[:max_tokens]
        x = x[keep]
    return x.contiguous()


@torch.no_grad()
def nearest_code(x: torch.Tensor, codebook: torch.Tensor, chunk_size: int) -> torch.Tensor:
    codes = []
    for start in range(0, x.shape[0], chunk_size):
        chunk = x[start : start + chunk_size]
        # Features are normalized by default, so dot product is cosine nearest.
        scores = chunk @ codebook.t()
        codes.append(scores.argmax(dim=-1).cpu())
    return torch.cat(codes, dim=0)


@torch.no_grad()
def fit_kmeans(
    x_cpu: torch.Tensor,
    codebook_size: int,
    iters: int,
    chunk_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    if x_cpu.shape[0] < codebook_size:
        raise RuntimeError(f"Need at least {codebook_size} tokens, got {x_cpu.shape[0]}")
    gen = torch.Generator().manual_seed(seed)
    init = torch.randperm(x_cpu.shape[0], generator=gen)[:codebook_size]
    codebook = x_cpu[init].to(device=device, dtype=torch.float32).contiguous()
    codebook = F.normalize(codebook, dim=-1)
    x_device = x_cpu.to(device=device, dtype=torch.float32)

    for _ in tqdm(range(iters), desc="kmeans"):
        sums = torch.zeros_like(codebook)
        counts = torch.zeros(codebook_size, device=device, dtype=torch.float32)
        for start in range(0, x_device.shape[0], chunk_size):
            chunk = x_device[start : start + chunk_size]
            idx = (chunk @ codebook.t()).argmax(dim=-1)
            sums.index_add_(0, idx, chunk)
            counts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        empty = counts == 0
        if empty.any():
            repl = torch.randint(0, x_device.shape[0], (int(empty.sum().item()),), device=device)
            sums[empty] = x_device[repl]
            counts[empty] = 1.0
        codebook = F.normalize(sums / counts.clamp_min(1.0).unsqueeze(1), dim=-1)
    return codebook.cpu()


def write_discrete_labels(
    paths: list[Path],
    output_dir: Path,
    codebook: torch.Tensor,
    normalize: bool,
    chunk_size: int,
    device: torch.device,
    overwrite: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    codebook_device = codebook.to(device=device, dtype=torch.float32)
    written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for path in tqdm(paths, desc="quantize labels"):
            payload, plan = load_plan(path)
            out_path = output_dir / path.name
            if out_path.exists() and not overwrite:
                rec = {"stem": payload["stem"], "path": str(out_path)}
                manifest.write(json.dumps(rec) + "\n")
                continue
            plan = maybe_normalize(plan, normalize).to(device=device, dtype=torch.float32)
            codes = nearest_code(plan, codebook_device, chunk_size=chunk_size)
            semantic_plan = payload["semantic_plan"]
            code_shape = tuple(semantic_plan.shape[:-1])
            code_payload = {
                "stem": payload["stem"],
                "video_path": payload.get("video_path"),
                "prompt": payload.get("prompt"),
                "first_frame_index": int(payload.get("first_frame_index", 0)),
                "future_frame_indices": payload.get("future_frame_indices", []),
                "semantic_token_ids": codes.reshape(code_shape).to(torch.long),
                "semantic_token_shape": code_shape,
                "codebook_size": int(codebook.shape[0]),
                "semantic_dim": int(codebook.shape[1]),
                "grid_size": int(payload.get("grid_size", code_shape[-1] if len(code_shape) > 1 else 0)),
                "num_keyframes": int(payload.get("num_keyframes", code_shape[0] if code_shape else 0)),
                "source_continuous_path": str(path),
                "feature_type": "vq_qwen3vl_image_tokens",
            }
            torch.save(code_payload, out_path)
            rec = {
                "stem": code_payload["stem"],
                "path": str(out_path),
                "shape": list(code_shape),
                "codebook_size": int(codebook.shape[0]),
            }
            manifest.write(json.dumps(rec) + "\n")
            written += 1
    return written


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = list_label_paths(args.continuous_label_dir, args.max_samples)
    train_tokens = collect_training_tokens(paths, args.max_train_tokens, args.normalize, args.seed)
    codebook = fit_kmeans(
        train_tokens,
        codebook_size=args.codebook_size,
        iters=args.kmeans_iters,
        chunk_size=args.chunk_size,
        device=device,
        seed=args.seed,
    )
    codebook_path = args.output_dir / "semantic_codebook.pt"
    torch.save(
        {
            "codebook": codebook,
            "codebook_size": args.codebook_size,
            "normalize": args.normalize,
            "source_label_dir": str(args.continuous_label_dir),
            "feature_type": "vq_qwen3vl_image_tokens",
        },
        codebook_path,
    )
    written = write_discrete_labels(
        paths,
        output_dir=args.output_dir,
        codebook=codebook,
        normalize=args.normalize,
        chunk_size=args.chunk_size,
        device=device,
        overwrite=args.overwrite,
    )
    summary = {
        "continuous_label_dir": str(args.continuous_label_dir),
        "output_dir": str(args.output_dir),
        "codebook_path": str(codebook_path),
        "num_inputs": len(paths),
        "written": written,
        "codebook_size": args.codebook_size,
        "max_train_tokens": int(train_tokens.shape[0]),
        "semantic_dim": int(codebook.shape[1]),
        "normalize": args.normalize,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
