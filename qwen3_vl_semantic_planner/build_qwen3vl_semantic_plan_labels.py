#!/usr/bin/env python3
"""Precompute Plan-X-style continuous semantic plan labels with Qwen3-VL.

This script uses plain Qwen3-VL-Instruct as the semantic tokenizer: future RGB
keyframes are encoded, image-token hidden states are pooled to a fixed spatial
grid, and saved as continuous semantic plan labels for a later planner
fine-tune.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from qwen3vl_wrapper import load_qwen3vl_model_and_processor, move_qwen_inputs_to_device, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-ranges", type=Path, default=None)
    parser.add_argument("--num-keyframes", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def load_frame_ranges(path: Path) -> list[tuple[str, int, int]]:
    data = json.loads(path.read_text())
    items: list[tuple[str, int, int]] = []
    if isinstance(data, dict):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = ((str(x.get("stem") or x.get("video_id") or x.get("id")), x) for x in data)
    else:
        raise TypeError(f"Unsupported frame_ranges type: {type(data)!r}")

    for stem, ranges in iterable:
        if isinstance(ranges, dict):
            start = int(ranges.get("start", ranges.get("frame_start", 0)))
            end = int(ranges.get("end", ranges.get("frame_end", start + 49)))
        elif isinstance(ranges, list) and ranges and isinstance(ranges[0], (list, tuple)):
            start, end = int(ranges[0][0]), int(ranges[0][1])
        elif isinstance(ranges, list) and len(ranges) >= 2:
            start, end = int(ranges[0]), int(ranges[1])
        else:
            continue
        if end > start:
            items.append((stem, start, end))
    return items


def read_meta(dataset_root: Path, stem: str) -> str:
    meta = dataset_root / "metas" / f"{stem}.txt"
    if not meta.exists():
        return "A robot manipulates the target object."
    text = meta.read_text(errors="ignore").strip()
    return " ".join(text.split())


def load_frames(video_path: Path, frame_indices: list[int]) -> list[Image.Image]:
    import decord

    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    n = len(vr)
    idx = [min(max(int(i), 0), n - 1) for i in frame_indices]
    batch = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(arr).convert("RGB") for arr in batch]


def sample_indices(start: int, end: int, video_len: int, num_future: int) -> tuple[int, list[int]]:
    first = min(max(start, 0), max(video_len - 1, 0))
    last = min(max(end - 1, first + 1), max(video_len - 1, 0))
    if num_future <= 1:
        return first, [last]
    # Future semantic labels exclude the input frame, like Plan-X predicts S1..Sk from S0.
    raw = torch.linspace(first + 1, last, steps=num_future)
    future = [min(max(int(round(x.item())), 0), max(video_len - 1, 0)) for x in raw]
    return first, future


def get_video_len(video_path: Path) -> int:
    import decord

    return len(decord.VideoReader(str(video_path), ctx=decord.cpu(0)))


def get_torch_dtype(name: str) -> torch.dtype:
    return torch_dtype_from_name(name)


def make_chat_text(processor: Any) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Encode this robot manipulation frame as spatial semantic tokens.",
                },
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def encode_images(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    grid_size: int,
    device: torch.device,
    store_dtype: torch.dtype,
) -> torch.Tensor:
    text = [make_chat_text(processor) for _ in images]
    inputs = processor(text=text, images=images, padding=True, return_tensors="pt")
    model_dtype = next(model.parameters()).dtype
    inputs = move_qwen_inputs_to_device(inputs, device, model_dtype=model_dtype)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[-1]
    input_ids = inputs["input_ids"]
    image_token_id = int(model.config.image_token_id)
    merge = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    grids = inputs.get("image_grid_thw")
    labels: list[torch.Tensor] = []

    for b in range(input_ids.shape[0]):
        mask = input_ids[b] == image_token_id
        tokens = hidden[b, mask].float()
        if tokens.numel() == 0:
            raise RuntimeError("No Qwen3-VL image tokens found in processor output.")

        if grids is not None and b < grids.shape[0]:
            _, gh, gw = [int(x) for x in grids[b].tolist()]
            h = max(1, gh // merge)
            w = max(1, gw // merge)
        else:
            side = int(math.sqrt(tokens.shape[0]))
            h = w = side
        expected = h * w
        if expected != tokens.shape[0]:
            side = int(math.sqrt(tokens.shape[0]))
            if side * side != tokens.shape[0]:
                raise RuntimeError(f"Cannot infer image token grid from {tokens.shape[0]} tokens.")
            h = w = side

        grid = tokens.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(grid, (grid_size, grid_size))
        labels.append(pooled.squeeze(0).permute(1, 2, 0).reshape(grid_size * grid_size, -1))

    return torch.stack(labels, dim=0).to(dtype=store_dtype).cpu()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    progress_path = args.output_dir / "progress.jsonl"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch_dtype = get_torch_dtype(args.dtype)
    model, processor = load_qwen3vl_model_and_processor(
        args.model_path,
        device=device,
        dtype=args.dtype,
        attn_implementation="sdpa",
        local_files_only=True,
        eval_mode=True,
    )

    frame_ranges = args.frame_ranges or (args.dataset_root / "frame_ranges.json")
    items = load_frame_ranges(frame_ranges)
    if args.start_index:
        items = items[args.start_index :]
    if args.max_samples > 0:
        items = items[: args.max_samples]

    written = 0
    with manifest_path.open("a", encoding="utf-8") as manifest, progress_path.open("a", encoding="utf-8") as progress:
        for idx, (stem, start, end) in enumerate(tqdm(items, desc="qwen3vl semantic labels")):
            out_path = args.output_dir / f"{stem}.pt"
            if out_path.exists() and not args.overwrite:
                continue
            video_path = args.dataset_root / "videos" / f"{stem}.mp4"
            if not video_path.exists():
                continue
            try:
                video_len = get_video_len(video_path)
                first_idx, future_indices = sample_indices(start, end, video_len, args.num_keyframes)
                frames = load_frames(video_path, future_indices)
                semantic_plan = encode_images(
                    model=model,
                    processor=processor,
                    images=frames,
                    grid_size=args.grid_size,
                    device=device,
                    store_dtype=torch_dtype,
                )
                payload = {
                    "stem": stem,
                    "video_path": str(video_path),
                    "prompt": read_meta(args.dataset_root, stem),
                    "first_frame_index": first_idx,
                    "future_frame_indices": future_indices,
                    "semantic_plan": semantic_plan,
                    "semantic_plan_shape": tuple(semantic_plan.shape),
                    "semantic_dim": int(semantic_plan.shape[-1]),
                    "grid_size": args.grid_size,
                    "num_keyframes": args.num_keyframes,
                    "model_path": str(args.model_path),
                    "feature_type": "qwen3vl_last_hidden_image_tokens_pooled",
                }
                torch.save(payload, out_path)
                rec = {
                    "stem": stem,
                    "path": str(out_path),
                    "shape": list(semantic_plan.shape),
                    "first_frame_index": first_idx,
                    "future_frame_indices": future_indices,
                }
                manifest.write(json.dumps(rec) + "\n")
                manifest.flush()
                written += 1
                if written % max(args.log_every, 1) == 0:
                    progress.write(json.dumps({"written": written, "last": stem}) + "\n")
                    progress.flush()
            except Exception as exc:  # keep long precompute jobs resilient
                progress.write(json.dumps({"stem": stem, "error": repr(exc)}) + "\n")
                progress.flush()

    summary = {
        "dataset_root": str(args.dataset_root),
        "model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "num_keyframes": args.num_keyframes,
        "grid_size": args.grid_size,
        "written_this_run": written,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
