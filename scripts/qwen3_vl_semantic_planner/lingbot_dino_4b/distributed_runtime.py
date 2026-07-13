from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager


@dataclass(frozen=True)
class RuntimeContract:
    distributed_type: str
    world_size: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    zero_stage: int | None


def _distributed_type_name(accelerator: Any) -> str:
    value = getattr(accelerator, "distributed_type", "NO")
    name = getattr(value, "name", str(value))
    return name.rsplit(".", 1)[-1].upper()


def is_deepspeed(accelerator: Any) -> bool:
    return _distributed_type_name(accelerator) == "DEEPSPEED"


def _deepspeed_config(accelerator: Any) -> dict[str, Any]:
    plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    config = getattr(plugin, "deepspeed_config", None)
    if plugin is None or not isinstance(config, dict):
        raise RuntimeError("DeepSpeed runtime has no resolved DeepSpeed configuration")
    return config


def validate_runtime_contract(
    accelerator: Any,
    *,
    per_device_batch_size: int,
    grad_accum: int,
    expected_global_batch: int,
) -> RuntimeContract:
    if per_device_batch_size <= 0 or grad_accum <= 0:
        raise RuntimeError("batch size and gradient accumulation must be positive")
    world_size = int(accelerator.num_processes)
    accelerator_accum = int(accelerator.gradient_accumulation_steps)
    if accelerator_accum != grad_accum:
        raise RuntimeError(
            "Accelerator gradient accumulation mismatch: "
            f"trainer={grad_accum}, accelerator={accelerator_accum}"
        )
    global_batch = world_size * per_device_batch_size * grad_accum
    if expected_global_batch > 0 and global_batch != expected_global_batch:
        raise RuntimeError(
            f"global batch mismatch: computed={global_batch}, "
            f"expected={expected_global_batch}"
        )
    zero_stage = None
    if is_deepspeed(accelerator):
        config = _deepspeed_config(accelerator)
        deepspeed_accum = int(config.get("gradient_accumulation_steps", -1))
        if deepspeed_accum != grad_accum:
            raise RuntimeError(
                "DeepSpeed gradient accumulation mismatch: "
                f"trainer={grad_accum}, deepspeed={deepspeed_accum}"
            )
        zero_stage = int(config.get("zero_optimization", {}).get("stage", -1))
        if zero_stage != 2:
            raise RuntimeError(f"ZeRO stage mismatch: expected 2, got {zero_stage}")
    return RuntimeContract(
        distributed_type=_distributed_type_name(accelerator),
        world_size=world_size,
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        global_batch_size=global_batch,
        zero_stage=zero_stage,
    )


def build_accelerator(*, grad_accum: int, dtype: str):
    from accelerate import Accelerator

    mixed_precision = {"bf16": "bf16", "fp16": "fp16", "fp32": "no"}[dtype]
    return Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=mixed_precision,
    )


def accumulation_context(accelerator: Any, model: Any) -> ContextManager[None]:
    if is_deepspeed(accelerator):
        return nullcontext()
    return accelerator.accumulate(model)


def is_optimizer_update(accelerator: Any, micro_step: int, grad_accum: int) -> bool:
    if is_deepspeed(accelerator):
        return micro_step % grad_accum == 0
    return bool(accelerator.sync_gradients)


def checkpoint_module(accelerator: Any, model: Any) -> Any:
    return accelerator.unwrap_model(model)
