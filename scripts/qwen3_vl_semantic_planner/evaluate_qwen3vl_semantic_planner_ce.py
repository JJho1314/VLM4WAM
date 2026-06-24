#!/usr/bin/env python3
"""Evaluate a Qwen3-VL Stage-A semantic-token planner checkpoint.

The training loss is teacher-forced.  This script checks whether the saved
planner can freely emit a usable fixed-length semantic token plan:

- valid semantic-token ratio
- generated length against the expected K * G * G plan length
- positional token accuracy against the VQ labels
- code-set overlap and repetition/collapse statistics
- optional teacher-forced loss/accuracy on the same sampled episodes
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from qwen3vl_wrapper import move_qwen_inputs_to_device, torch_dtype_from_name
from train_qwen3vl_semantic_planner_ce import SEM_TOKEN_TEMPLATE


SEM_TOKEN_SEPARATORS = {"none": "", "space": " "}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--discrete-label-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--num-keyframes", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--teacher-forced", action="store_true")
    parser.add_argument("--constrain-semantic-vocab", action="store_true")
    parser.add_argument("--semantic-token-separator", choices=["auto", "none", "space"], default="auto")
    return parser.parse_args()


def semantic_token(code_id: int) -> str:
    return SEM_TOKEN_TEMPLATE.format(int(code_id))


def join_semantic_tokens(tokens: list[str], separator: str) -> str:
    return SEM_TOKEN_SEPARATORS[separator].join(tokens)


def load_manifest(label_dir: Path) -> list[Path]:
    manifest = label_dir / "manifest.jsonl"
    if manifest.exists():
        paths: list[Path] = []
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            path = Path(rec["path"])
            if path.exists():
                paths.append(path)
        return paths
    return [p for p in sorted(label_dir.glob("*.pt")) if p.name != "semantic_codebook.pt"]


def load_frame(video_path: Path, index: int) -> Image.Image:
    import decord

    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    index = min(max(int(index), 0), len(vr) - 1)
    return Image.fromarray(vr[index].asnumpy()).convert("RGB")


def planner_instruction(prompt: str, expected_tokens: int, separator: str) -> str:
    if separator == "none":
        output_format = (
            f"Output exactly {expected_tokens} consecutive semantic tokens with no spaces, "
            "newlines, punctuation, or natural language."
        )
    else:
        output_format = (
            f"Output exactly {expected_tokens} semantic tokens separated by single spaces, "
            "with no other text."
        )
    return (
        "You are a robot video semantic planner. Given the first frame and instruction, "
        "predict the future spatial semantic token sequence for the manipulation video. "
        f"{output_format}\n"
        f"Instruction: {' '.join(str(prompt).split())}"
    )


def make_generation_text(processor: Any, prompt: str, expected_tokens: int, separator: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": planner_instruction(prompt, expected_tokens, separator)},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_teacher_text(processor: Any, prompt: str, gt_codes: torch.Tensor, separator: str) -> str:
    tokens = [semantic_token(int(x)) for x in gt_codes.reshape(-1).tolist()]
    assistant_text = join_semantic_tokens(tokens, separator)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": planner_instruction(prompt, len(tokens), separator)},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def resolve_separator(args: argparse.Namespace) -> str:
    if args.semantic_token_separator != "auto":
        return args.semantic_token_separator
    meta_path = args.checkpoint_dir / "planner_meta.json"
    if not meta_path.exists():
        return "space"
    meta = json.loads(meta_path.read_text())
    return str(meta.get("semantic_token_separator", "space"))


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, list[int], dict[int, int]]:
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.checkpoint_dir / "processor", local_files_only=True)
    processor.tokenizer.padding_side = "left"

    meta = json.loads((args.checkpoint_dir / "planner_meta.json").read_text())
    codebook_size = int(meta["codebook_size"])
    sem_tokens = [semantic_token(i) for i in range(codebook_size)]
    sem_token_ids = [int(processor.tokenizer.convert_tokens_to_ids(tok)) for tok in sem_tokens]
    if any(x < 0 for x in sem_token_ids):
        raise RuntimeError("Some semantic tokens are missing from the checkpoint tokenizer.")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch_dtype_from_name(args.dtype),
        attn_implementation="sdpa",
        local_files_only=True,
    )
    try:
        model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(len(processor.tokenizer))

    sem_weight_path = args.checkpoint_dir / "semantic_token_weights.pt"
    if sem_weight_path.exists():
        state = torch.load(sem_weight_path, map_location="cpu", weights_only=False)
        token_ids = [int(x) for x in state["semantic_token_ids"]]
        with torch.no_grad():
            input_emb = model.get_input_embeddings()
            input_emb.weight[token_ids] = state["input_embeddings"].to(dtype=input_emb.weight.dtype)
            output_emb = model.get_output_embeddings()
            if output_emb is not None and "output_embeddings" in state:
                output_emb.weight[token_ids] = state["output_embeddings"].to(dtype=output_emb.weight.dtype)

    model = PeftModel.from_pretrained(model, args.checkpoint_dir / "qwen3vl_lora_or_model")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, sem_token_ids, {tok_id: i for i, tok_id in enumerate(sem_token_ids)}


def parse_semantic_codes(new_token_ids: torch.Tensor, sem_id_to_code: dict[int, int]) -> list[int]:
    out: list[int] = []
    for token_id in new_token_ids.tolist():
        code = sem_id_to_code.get(int(token_id))
        if code is not None:
            out.append(code)
    return out


class SemanticOnlyLogitsProcessor:
    def __init__(self, semantic_token_ids: list[int]) -> None:
        self.semantic_token_ids = semantic_token_ids
        self._mask_cache: dict[tuple[torch.device, int], torch.Tensor] = {}

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        key = (scores.device, scores.shape[-1])
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = torch.full((scores.shape[-1],), float("-inf"), device=scores.device, dtype=scores.dtype)
            valid = torch.tensor(self.semantic_token_ids, device=scores.device, dtype=torch.long)
            mask[valid] = 0
            self._mask_cache[key] = mask
        return scores + mask


def positional_accuracy(pred: list[int], gt: torch.Tensor, expected_len: int) -> float:
    gt_list = [int(x) for x in gt.reshape(-1).tolist()]
    matches = 0
    for i in range(min(len(pred), expected_len)):
        if pred[i] == gt_list[i]:
            matches += 1
    return matches / float(expected_len)


def code_set_iou(pred: list[int], gt: torch.Tensor) -> float:
    pred_set = set(pred)
    gt_set = set(int(x) for x in gt.reshape(-1).tolist())
    if not pred_set and not gt_set:
        return 1.0
    union = pred_set | gt_set
    return len(pred_set & gt_set) / float(len(union)) if union else 0.0


def gt_coverage(pred: list[int], gt: torch.Tensor) -> float:
    pred_set = set(pred)
    gt_set = set(int(x) for x in gt.reshape(-1).tolist())
    return len(pred_set & gt_set) / float(len(gt_set)) if gt_set else 0.0


def adjacent_repeat_fraction(pred: list[int]) -> float:
    if len(pred) <= 1:
        return 0.0
    repeats = sum(1 for a, b in zip(pred[:-1], pred[1:]) if a == b)
    return repeats / float(len(pred) - 1)


def most_common_fraction(pred: list[int]) -> float:
    if not pred:
        return 0.0
    return collections.Counter(pred).most_common(1)[0][1] / float(len(pred))


@torch.no_grad()
def teacher_forced_metrics(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    gt_codes: torch.Tensor,
    sem_token_ids: list[int],
    device: torch.device,
    separator: str,
) -> tuple[float, float]:
    text = make_teacher_text(processor, prompt, gt_codes, separator)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    labels = torch.full_like(inputs["input_ids"], -100)
    sem_ids = torch.tensor(sem_token_ids, dtype=inputs["input_ids"].dtype)
    sem_mask = torch.isin(inputs["input_ids"][0], sem_ids)
    labels[0, sem_mask] = inputs["input_ids"][0, sem_mask]
    inputs["labels"] = labels
    model_dtype = next(model.parameters()).dtype
    inputs = move_qwen_inputs_to_device(inputs, device, model_dtype=model_dtype)
    out = model(**inputs, use_cache=False)
    shift_logits = out.logits[:, :-1, :]
    shift_labels = inputs["labels"][:, 1:]
    valid = shift_labels != -100
    if bool(valid.any()):
        pred = shift_logits.argmax(dim=-1)
        acc = float((pred[valid] == shift_labels[valid]).float().mean().item())
    else:
        acc = 0.0
    return float(out.loss.detach().float().item()), acc


@torch.no_grad()
def evaluate_one(
    model: Any,
    processor: Any,
    path: Path,
    args: argparse.Namespace,
    sem_token_ids: list[int],
    sem_id_to_code: dict[int, int],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stem = payload["stem"]
    video_path = Path(payload.get("video_path") or args.dataset_root / "videos" / f"{stem}.mp4")
    prompt = payload.get("prompt") or "A robot manipulates the target object."
    first_frame_index = int(payload.get("first_frame_index", 0))
    gt_codes = payload["semantic_token_ids"].long().reshape(-1)
    expected_len = args.num_keyframes * args.grid_size * args.grid_size
    separator = resolve_separator(args)
    image = load_frame(video_path, first_frame_index)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    text = make_generation_text(processor, prompt, expected_len, separator)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[1])
    model_dtype = next(model.parameters()).dtype
    inputs = move_qwen_inputs_to_device(inputs, device, model_dtype=model_dtype)
    gen = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens or expected_len,
        do_sample=False,
        use_cache=True,
        logits_processor=[SemanticOnlyLogitsProcessor(sem_token_ids)] if args.constrain_semantic_vocab else None,
    )
    new_ids = gen[0, input_len:].detach().cpu()
    pred_codes = parse_semantic_codes(new_ids, sem_id_to_code)
    total_new = int(new_ids.numel())

    rec: dict[str, Any] = {
        "stem": stem,
        "prompt": prompt,
        "generated_new_tokens": total_new,
        "valid_semantic_tokens": len(pred_codes),
        "expected_semantic_tokens": expected_len,
        "constrained_semantic_vocab": bool(args.constrain_semantic_vocab),
        "semantic_token_separator": separator,
        "valid_ratio_over_generated": len(pred_codes) / float(max(total_new, 1)),
        "length_ratio_over_expected": len(pred_codes) / float(expected_len),
        "positional_token_acc": positional_accuracy(pred_codes, gt_codes, expected_len),
        "code_set_iou": code_set_iou(pred_codes, gt_codes),
        "gt_code_coverage": gt_coverage(pred_codes, gt_codes),
        "unique_pred_codes": len(set(pred_codes)),
        "unique_gt_codes": len(set(int(x) for x in gt_codes.tolist())),
        "unique_ratio": len(set(pred_codes)) / float(max(len(pred_codes), 1)),
        "most_common_frac": most_common_fraction(pred_codes),
        "adjacent_repeat_frac": adjacent_repeat_fraction(pred_codes),
        "pred_prefix": pred_codes[:24],
        "gt_prefix": [int(x) for x in gt_codes[:24].tolist()],
    }
    if args.teacher_forced:
        loss, acc = teacher_forced_metrics(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            gt_codes=gt_codes,
            sem_token_ids=sem_token_ids,
            device=device,
            separator=separator,
        )
        rec["teacher_forced_loss"] = loss
        rec["teacher_forced_acc"] = acc
    return rec


def mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    paths = load_manifest(args.discrete_label_dir)
    random.shuffle(paths)
    paths = paths[: args.num_samples]
    model, processor, sem_token_ids, sem_id_to_code = load_model_and_processor(args)
    separator = resolve_separator(args)

    records = []
    for path in tqdm(paths, desc="evaluate semantic planner"):
        records.append(evaluate_one(model, processor, path, args, sem_token_ids, sem_id_to_code))

    metric_keys = [
        "generated_new_tokens",
        "valid_semantic_tokens",
        "valid_ratio_over_generated",
        "length_ratio_over_expected",
        "positional_token_acc",
        "code_set_iou",
        "gt_code_coverage",
        "unique_pred_codes",
        "unique_gt_codes",
        "unique_ratio",
        "most_common_frac",
        "adjacent_repeat_frac",
        "teacher_forced_loss",
        "teacher_forced_acc",
    ]
    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "discrete_label_dir": str(args.discrete_label_dir),
        "num_samples": len(records),
        "expected_semantic_tokens": args.num_keyframes * args.grid_size * args.grid_size,
        "constrained_semantic_vocab": bool(args.constrain_semantic_vocab),
        "semantic_token_separator": separator,
        "metrics_mean": {
            key: mean([float(r[key]) for r in records if key in r])
            for key in metric_keys
            if any(key in r for r in records)
        },
    }
    args.output_json.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
