"""Inference-time loader for a trained CoVT semantic planner.

Keystone for Stage-2 joint training: loads a planner checkpoint (produced by
``train_qwen3vl_semantic_planner.py``'s ``save_checkpoint``) and exposes a single
``predict(images, prompts) -> [B, target_len, semantic_dim]`` call that returns the
predicted SigLIP semantic plan in the *same* raw-feature space the Cosmos world model
consumes under ``data_batch["semantic_plan"]``.

It is reused three ways:
  * offline plan generation  (``gen_predicted_plans.py``, Tier-1a)
  * in-loop frozen planner    (WM online-encoder hook, no_grad → Tier-1b)
  * in-loop trainable planner  (same hook, grad on + throttle → Tier-2 e2e)

Faithful to the *current* distinct-token CoVT interface: it reuses ``PlannerWrapper`` /
``CoVTLatentDecoderHead`` from the training module and reconstructs the assistant
plan-token sequence exactly as ``Collator`` does, so predictions are bit-identical to
what training/eval would produce for the same checkpoint.

NOTE the stale ``evaluate_qwen3vl_semantic_planner_regression.py`` uses the OLD single
``plan_token_id`` interface and does NOT work with covt checkpoints — do not reuse it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from PIL import Image

from qwen3vl_wrapper import load_qwen3vl_model_and_processor, move_qwen_inputs_to_device
from train_qwen3vl_semantic_planner import PlannerWrapper

# Same user-turn wording the trainer/Collator uses; predictions depend on it verbatim.
_USER_TEMPLATE = (
    "You are a robot video semantic planner. Given the first frame and instruction, "
    "predict future spatial semantic plan tokens for the manipulation video.\n"
    "Instruction: {prompt}"
)


def _plan_sequence_from_meta(meta: dict[str, Any]) -> list[str]:
    """Reconstruct the assistant plan-token string list exactly as main() built it."""
    head = str(meta.get("plan_head_type", "mlp"))
    if head == "covt":
        latent_len = int(meta["latent_len"])  # num_keyframes * num_latent_per_keyframe
        return [f"<|sem_plan_{i}|>" for i in range(latent_len)]
    # mlp / baton_crossattn: a single repeated <|sem_plan|> filling the dense grid
    n = int(meta["num_keyframes"]) * int(meta["grid_size"]) * int(meta["grid_size"])
    return ["<|sem_plan|>"] * n


class PlannerPlanProvider(nn.Module):
    """Load a trained planner checkpoint and predict semantic plans from (image, prompt)."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: torch.device | str = "cuda",
        dtype: str = "bf16",
        trainable: bool = False,
        model_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        ckpt = Path(checkpoint_dir)
        meta = json.loads((ckpt / "planner_meta.json").read_text())
        self.meta = meta
        self.device = torch.device(device)
        self.target_len = int(meta["target_len"])
        self.semantic_dim = int(meta["semantic_dim"])
        self.plan_sequence = _plan_sequence_from_meta(meta)

        # Processor (with the added <|sem_plan_i|> tokens) is saved inside the checkpoint, so we
        # never need the original base-model path for full-FT — important on a box where
        # meta["model_path"] (the HPC3 training path) does not exist.
        from transformers import AutoProcessor

        proc_dir = ckpt / "processor"
        self.processor = AutoProcessor.from_pretrained(
            proc_dir if proc_dir.exists() else (model_path or meta["model_path"]),
            local_files_only=True,
        )

        # Full-FT checkpoints save the whole resized model (with trained plan-token rows) under
        # qwen3vl_lora_or_model/ — load it directly. LoRA checkpoints save an adapter over a base
        # model (then model_path, or meta["model_path"], must resolve on this box).
        adapter_dir = ckpt / "qwen3vl_lora_or_model"
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        if (adapter_dir / "adapter_config.json").exists():
            from peft import PeftModel

            base_model, _ = load_qwen3vl_model_and_processor(
                model_path or meta["model_path"], device=None, dtype=dtype,
                attn_implementation="sdpa", local_files_only=True, eval_mode=not trainable,
            )
            base_model.resize_token_embeddings(len(self.processor.tokenizer))
            model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=trainable)
            emb_path = ckpt / "plan_token_embedding.pt"
            if emb_path.exists():
                vec = torch.load(emb_path, map_location="cpu", weights_only=False)
                ids = torch.as_tensor(meta["plan_token_ids"])
                w = model.get_input_embeddings().weight
                w.data[ids] = vec.to(dtype=w.dtype)
        else:
            from transformers import Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                adapter_dir, torch_dtype=torch_dtype, local_files_only=True,
                attn_implementation="sdpa",
            )

        hidden_size = int(model.config.text_config.hidden_size)
        wrapper = PlannerWrapper(
            model=model,
            hidden_size=hidden_size,
            semantic_dim=self.semantic_dim,
            plan_token_ids=[int(x) for x in meta["plan_token_ids"]],
            target_len=self.target_len,
            num_keyframes=int(meta["num_keyframes"]),
            grid_size=int(meta["grid_size"]),
            num_latent_per_keyframe=int(meta.get("num_latent_per_keyframe", 4)),
            plan_head_type=str(meta.get("plan_head_type", "mlp")),
            plan_head_num_heads=int(meta.get("plan_head_num_heads", 16)),
            plan_head_dropout=float(meta.get("plan_head_dropout", 0.0)),
            sem_mlp_hidden_size=int(meta.get("sem_mlp_hidden_size", 0)),
        )
        state = torch.load(ckpt / "plan_head.pt", map_location="cpu", weights_only=False)
        wrapper.plan_head.load_state_dict(state)
        wrapper.to(self.device)
        wrapper.eval() if not trainable else wrapper.train()
        wrapper.requires_grad_(trainable)
        self.wrapper = wrapper

    def build_inputs(self, images: Sequence[Image.Image], prompts: Sequence[str]) -> dict[str, Any]:
        """Replicate Collator: user(image+instruction) + assistant(plan tokens), teacher-forced."""
        plan_text = " ".join(self.plan_sequence)
        texts = []
        for prompt in prompts:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": _USER_TEMPLATE.format(prompt=prompt)},
                ]},
                {"role": "assistant", "content": plan_text},
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        inputs = self.processor(text=texts, images=list(images), padding=True, return_tensors="pt")
        return inputs

    def predict(self, images: Sequence[Image.Image], prompts: Sequence[str]) -> torch.Tensor:
        """[B, target_len(=3645), semantic_dim(=1152)] predicted plan, float32, on self.device.

        Differentiable when ``trainable=True`` (Tier-2). Wrap the call in ``torch.no_grad()``
        for Tier-1 frozen-planner use.
        """
        inputs = self.build_inputs(images, prompts)
        model_dtype = next(self.wrapper.model.parameters()).dtype
        inputs = move_qwen_inputs_to_device(inputs, self.device, model_dtype=model_dtype)
        return self.wrapper.predict_semantic_plan(**inputs)

    @torch.no_grad()
    def predict_frozen(self, images: Sequence[Image.Image], prompts: Sequence[str]) -> torch.Tensor:
        return self.predict(images, prompts)
