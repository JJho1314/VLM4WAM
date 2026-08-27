#!/usr/bin/env python3
"""Extract the Qwen3-VL-4B VLM out of a lingbot-vla-v2-6b checkpoint into a STOCK HF checkpoint.

The 6b release nests a genuine Qwen3-VL-4B under ``model.qwenvl_with_expert.qwenvl.*`` (verified from
the real safetensors index: 713 tensors — text has ``self_attn.q_norm/k_norm`` = Qwen3 QK-norm, visual
has ``deepstack``+``merger`` = Qwen3-VL vision tower). This script selects those tensors, strips the
prefix so the names become stock ``model.language_model.*`` / ``model.visual.*``, and writes a checkpoint
that loads with a plain ``Qwen3VLForConditionalGeneration``. The action expert (``qwen_expert.*``), the
four align heads, and the action projections are intentionally dropped — the align-head warm-start is
handled separately by the 4B training script (it reads ``future_video_align_head.*`` from the same src).

Usage:
    python extract_qwenvl_from_lingbot.py \
        --src   /path/to/lingbot-vla-v2-6b            # dir with the 6 shards + index.json
        --ref-4b /path/to/Qwen3-VL-4B-Instruct        # a REAL 4B dir: config/tokenizer/preprocessor
        --out   /path/to/Qwen3-VL-4B-lingbot-vlm      # output (feed this to planner --model-path)

Notes:
  * lingbot's own config.json is a 31-byte ``{"vlm_family":"qwen3_vl"}`` stub, so config/tokenizer MUST
    come from a real Qwen3-VL-4B-Instruct (``--ref-4b``).
  * vocab_size is auto-detected from the extracted embed_tokens (lingbot may have resized the vocab); the
    ref config is overridden to match so the load is shape-clean.
  * lm_head is absent on disk (tied embeddings); a tied-embeddings config reconstructs it on load.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "model.qwenvl_with_expert.qwenvl."
# tokenizer/processor/config files copied verbatim from the reference 4B (NOT the model weights)
_REF_COPY = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "added_tokens.json", "special_tokens_map.json", "chat_template.json",
    "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="lingbot-vla-v2-6b dir (shards + index.json)")
    ap.add_argument("--ref-4b", required=True, type=Path, help="real Qwen3-VL-4B-Instruct dir")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-check", action="store_true", help="skip the trial from_pretrained self-check")
    args = ap.parse_args()

    src: Path = args.src
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    want = {k: v for k, v in weight_map.items() if k.startswith(PREFIX)}
    if not want:
        raise RuntimeError(f"no keys under prefix {PREFIX!r} — is this a lingbot-vla-v2 6b checkpoint?")
    by_file: dict[str, list[str]] = defaultdict(list)
    for k, f in want.items():
        by_file[f].append(k)
    print(f"[extract] {len(want)} VLM tensors across {len(by_file)} shard(s)")

    new_state: dict[str, torch.Tensor] = {}
    vocab_size = None
    for fname, keys in by_file.items():
        with safe_open(src / fname, framework="pt", device="cpu") as sf:
            for k in keys:
                nk = k[len(PREFIX):]  # -> model.language_model.* / model.visual.*
                t = sf.get_tensor(k)
                new_state[nk] = t
                if nk.endswith("language_model.embed_tokens.weight"):
                    vocab_size = int(t.shape[0])
    print(f"[extract] stripped prefix; embed_tokens vocab_size = {vocab_size}")

    # sanity: expect stock Qwen3-VL submodule roots
    roots = sorted({".".join(k.split(".")[:2]) for k in new_state})
    print(f"[extract] top-level roots: {roots}")
    assert any(r.startswith("model.language_model") for r in roots), "no language_model.* keys after strip"
    assert any(r.startswith("model.visual") for r in roots), "no visual.* keys after strip"

    # --- config + tokenizer/processor from the reference 4B, vocab overridden ---
    ref: Path = args.ref_4b
    cfg = json.loads((ref / "config.json").read_text())
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
        tc = cfg.get("text_config")
        if isinstance(tc, dict):
            tc["vocab_size"] = vocab_size
    cfg["tie_word_embeddings"] = True  # lm_head absent on disk -> tie to embed_tokens
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for name in _REF_COPY:
        srcf = ref / name
        if srcf.exists():
            shutil.copy2(srcf, out / name)
    print(f"[extract] copied config (+vocab override) and tokenizer/processor from {ref}")

    # --- save weights (single safetensors; ~8GB for a 4B bf16 model fits comfortably) ---
    new_state = {k: (v.contiguous()) for k, v in new_state.items()}
    save_file(new_state, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"[extract] wrote {out/'model.safetensors'} ({len(new_state)} tensors)")

    # --- self-check: does it load as a stock Qwen3-VL? ---
    if not args.no_check:
        try:
            from transformers import Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                out, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa",
            )
            n = sum(p.numel() for p in model.parameters())
            print(f"[check] OK — loaded stock Qwen3-VL, {n/1e9:.2f}B params")
        except Exception as e:  # noqa: BLE001
            print(f"[check] load reported: {type(e).__name__}: {e}")
            print("[check] inspect the missing/unexpected keys above; if only lm_head is 'missing' that is "
                  "expected (tied). Re-run with --no-check once satisfied.")


if __name__ == "__main__":
    main()
