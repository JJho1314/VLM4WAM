# LingBot Planner ZeRO-2 Training and Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the current 4 × 64-query Qwen3-VL 4B LingBot planner through Accelerate + DeepSpeed ZeRO-2 and replace seven experiment-specific launchers with one generic launcher plus POD and HPC3 profiles.

**Architecture:** Preserve the current planner, online DINO/MoGe/MoRGBD teachers, four losses, and FastWAM export format. Add a small distributed-runtime module and a deterministic Accelerate/DeepSpeed config generator, then adapt the explicit training loop to `accelerator.prepare` and `accelerator.backward`. Keep only machine-specific path/resource settings in two thin profiles.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Accelerate, DeepSpeed ZeRO-2, pytest, Bash, Slurm, eight H100 GPUs.

## Global Constraints

- Input is one current frame plus one future frame at offset 8 from a nine-frame sample.
- Query order is current DINO, future DINO, current depth, future depth.
- Each query group has 64 private VLM tokens; total VLM task-token length is 256.
- Every prediction head emits 256 × 1024 teacher features.
- Current/future DINO and current/future depth loss weights are all 0.004.
- Full language-model fine-tuning remains enabled; vision tower and LM head remain frozen.
- Training uses BF16, SDPA, no gradient checkpointing, and 12,000 optimizer steps.
- Preferred launch is eight GPUs × batch 8/GPU × accumulation 2 = global batch 128.
- ZeRO stage 2 uses no CPU offload and gradient clipping 1.0.
- Existing FastWAM checkpoint names, files, metadata, and provider compatibility must not change.
- Preserve unrelated dirty-worktree changes. Do not create implementation commits that would absorb pre-existing user edits; use focused diffs and verification checkpoints instead.
- Do not stop or mutate the currently running POD training process.

---

## File structure

- Create `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py`: Accelerate construction, DeepSpeed detection, runtime-contract validation, accumulation context, and optimizer-boundary logic.
- Create `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py`: deterministic matched DeepSpeed JSON and Accelerate YAML generation.
- Modify `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`: replace manual DDP with the new runtime while preserving model/data/loss/export behavior.
- Modify `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh`: make it the only generic current-run launcher.
- Create `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh`: POD paths and direct-launch defaults only.
- Create `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch`: HPC3 paths and Slurm resources only.
- Create `tests/test_lingbot_zero2_runtime.py`: focused runtime/config tests.
- Modify `tests/test_lingbot_dino_depth_contract.py` and `tests/test_lingbot_k1_current_future.py`: retain behavioral tests and remove assertions tied only to deleted wrappers.
- Modify `scripts/qwen3_vl_semantic_planner/README.md`: document the canonical launch path and current 4 × 64 contract.
- Delete the seven wrappers listed in Task 5.

---

### Task 1: Add a testable Accelerate/ZeRO-2 runtime contract

**Files:**
- Create: `tests/test_lingbot_zero2_runtime.py`
- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py`

**Interfaces:**
- Produces: `RuntimeContract`, `build_accelerator`, `validate_runtime_contract`, `is_deepspeed`, `accumulation_context`, `is_optimizer_update`, and `checkpoint_module`.
- Consumes later: trainer arguments `batch_size`, `grad_accum`, `expected_global_batch`, and `dtype`.

- [ ] **Step 1: Write failing runtime-contract and boundary tests**

Create `tests/test_lingbot_zero2_runtime.py` with these tests and helpers:

```python
from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
    / "distributed_runtime.py"
)


def load_runtime():
    spec = importlib.util.spec_from_file_location("lingbot_distributed_runtime", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakePlugin:
    def __init__(self, *, grad_accum: int = 2, zero_stage: int = 2):
        self.deepspeed_config = {
            "gradient_accumulation_steps": grad_accum,
            "zero_optimization": {"stage": zero_stage},
        }


class FakeState:
    def __init__(self, plugin=None):
        self.deepspeed_plugin = plugin


class FakeAccelerator:
    def __init__(
        self,
        *,
        distributed_type="DEEPSPEED",
        num_processes=8,
        grad_accum=2,
        plugin=None,
    ):
        self.distributed_type = distributed_type
        self.num_processes = num_processes
        self.gradient_accumulation_steps = grad_accum
        self.state = FakeState(plugin)
        self.sync_gradients = False
        self.accumulate_calls = 0

    @contextmanager
    def accumulate(self, _model):
        self.accumulate_calls += 1
        yield

    def unwrap_model(self, model):
        return model.inner


def test_runtime_contract_accepts_eight_by_eight_by_two():
    module = load_runtime()
    accelerator = FakeAccelerator(plugin=FakePlugin())

    contract = module.validate_runtime_contract(
        accelerator,
        per_device_batch_size=8,
        grad_accum=2,
        expected_global_batch=128,
    )

    assert contract.world_size == 8
    assert contract.global_batch_size == 128
    assert contract.distributed_type == "DEEPSPEED"
    assert contract.zero_stage == 2


@pytest.mark.parametrize(
    ("accelerator", "message"),
    [
        (FakeAccelerator(grad_accum=4, plugin=FakePlugin(grad_accum=2)), "Accelerator"),
        (FakeAccelerator(plugin=FakePlugin(grad_accum=4)), "DeepSpeed"),
        (FakeAccelerator(plugin=FakePlugin(zero_stage=3)), "ZeRO stage"),
    ],
)
def test_runtime_contract_rejects_mismatched_accumulation_or_stage(accelerator, message):
    module = load_runtime()
    with pytest.raises(RuntimeError, match=message):
        module.validate_runtime_contract(
            accelerator,
            per_device_batch_size=8,
            grad_accum=2,
            expected_global_batch=128,
        )


def test_runtime_contract_rejects_wrong_global_batch():
    module = load_runtime()
    accelerator = FakeAccelerator(plugin=FakePlugin())
    with pytest.raises(RuntimeError, match="global batch"):
        module.validate_runtime_contract(
            accelerator,
            per_device_batch_size=4,
            grad_accum=2,
            expected_global_batch=128,
        )


def test_deepspeed_boundary_is_driven_by_microstep_count():
    module = load_runtime()
    accelerator = FakeAccelerator(plugin=FakePlugin())
    assert [module.is_optimizer_update(accelerator, step, 2) for step in range(1, 5)] == [
        False,
        True,
        False,
        True,
    ]


def test_non_deepspeed_uses_accelerate_accumulation_and_sync_flag():
    module = load_runtime()
    accelerator = FakeAccelerator(
        distributed_type="MULTI_GPU",
        plugin=None,
    )
    model = object()
    with module.accumulation_context(accelerator, model):
        pass
    assert accelerator.accumulate_calls == 1
    accelerator.sync_gradients = True
    assert module.is_optimizer_update(accelerator, micro_step=1, grad_accum=2) is True


def test_deepspeed_bypasses_no_sync_and_unwraps_checkpoint_module():
    module = load_runtime()
    accelerator = FakeAccelerator(plugin=FakePlugin())
    model = type("Wrapped", (), {"inner": object()})()
    with module.accumulation_context(accelerator, model):
        pass
    assert accelerator.accumulate_calls == 0
    assert module.checkpoint_module(accelerator, model) is model.inner
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py
```

Expected: FAIL because `distributed_runtime.py` does not exist.

- [ ] **Step 3: Implement the minimal distributed runtime**

Create `distributed_runtime.py` with:

```python
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
```

- [ ] **Step 4: Verify GREEN and inspect the focused diff**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py
git diff --check -- tests/test_lingbot_zero2_runtime.py scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py
```

Expected: all runtime tests pass and `git diff --check` prints nothing.

---

### Task 2: Generate matched Accelerate and DeepSpeed configurations

**Files:**
- Modify: `tests/test_lingbot_zero2_runtime.py`
- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py`

**Interfaces:**
- Produces: `make_zero2_configs(grad_accum: int, num_processes: int, output_dir: Path) -> tuple[Path, Path]`.
- Consumes later: generic launcher environment values `GRAD_ACCUM`, `NUM_GPUS`, and `OUTPUT_DIR`.

- [ ] **Step 1: Add failing config-generation tests**

Append to `tests/test_lingbot_zero2_runtime.py`:

```python
import json

import yaml


CONFIG_GENERATOR = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
    / "make_zero2_config.py"
)


def load_config_generator():
    spec = importlib.util.spec_from_file_location("lingbot_zero2_config", CONFIG_GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_zero2_config_matches_batch_accumulation_and_process_count(tmp_path):
    module = load_config_generator()
    accelerate_path, deepspeed_path = module.make_zero2_configs(
        grad_accum=2,
        num_processes=8,
        output_dir=tmp_path,
    )

    deepspeed = json.loads(deepspeed_path.read_text())
    accelerate = yaml.safe_load(accelerate_path.read_text())
    assert deepspeed["gradient_accumulation_steps"] == 2
    assert deepspeed["zero_optimization"]["stage"] == 2
    assert deepspeed["zero_optimization"]["overlap_comm"] is True
    assert deepspeed["zero_optimization"]["contiguous_gradients"] is True
    assert deepspeed["bf16"]["enabled"] is True
    assert deepspeed["fp16"]["enabled"] is False
    assert accelerate["num_processes"] == 8
    assert accelerate["distributed_type"] == "DEEPSPEED"
    assert Path(accelerate["deepspeed_config"]["deepspeed_config_file"]) == deepspeed_path


@pytest.mark.parametrize(("grad_accum", "num_processes"), [(0, 8), (2, 0)])
def test_zero2_config_rejects_nonpositive_runtime_values(tmp_path, grad_accum, num_processes):
    module = load_config_generator()
    with pytest.raises(ValueError, match="positive"):
        module.make_zero2_configs(
            grad_accum=grad_accum,
            num_processes=num_processes,
            output_dir=tmp_path,
        )
```

- [ ] **Step 2: Run only the new tests and verify RED**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py -k zero2_config
```

Expected: FAIL because `make_zero2_config.py` does not exist.

- [ ] **Step 3: Implement deterministic config generation and CLI output**

Create `make_zero2_config.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def make_zero2_configs(
    *,
    grad_accum: int,
    num_processes: int,
    output_dir: Path,
) -> tuple[Path, Path]:
    if grad_accum <= 0 or num_processes <= 0:
        raise ValueError("gradient accumulation and process count must be positive")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"np{num_processes}_ga{grad_accum}_zero2"
    deepspeed_path = output_dir / f"deepspeed_{tag}.json"
    accelerate_path = output_dir / f"accelerate_{tag}.yaml"
    deepspeed_config = {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": grad_accum,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 500_000_000,
            "reduce_scatter": True,
            "reduce_bucket_size": 500_000_000,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
    }
    deepspeed_path.write_text(
        json.dumps(deepspeed_config, indent=2) + "\n",
        encoding="utf-8",
    )
    accelerate_path.write_text(
        "\n".join(
            [
                "compute_environment: LOCAL_MACHINE",
                "debug: false",
                "deepspeed_config:",
                f'  deepspeed_config_file: "{deepspeed_path}"',
                "  deepspeed_multinode_launcher: standard",
                "  zero3_init_flag: false",
                "distributed_type: DEEPSPEED",
                "downcast_bf16: 'no'",
                "machine_rank: 0",
                "main_training_function: main",
                "mixed_precision: bf16",
                "num_machines: 1",
                f"num_processes: {num_processes}",
                "rdzv_backend: static",
                "same_network: true",
                "use_cpu: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return accelerate_path, deepspeed_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad-accum", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    accelerate_path, _ = make_zero2_configs(
        grad_accum=args.grad_accum,
        num_processes=args.num_processes,
        output_dir=args.output_dir,
    )
    print(accelerate_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify config tests and CLI**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py -k zero2_config
python scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py --grad-accum 2 --num-processes 8 --output-dir /tmp/vlm4wam-zero2-config-test
```

Expected: tests pass; the CLI prints an Accelerate YAML path under `/tmp/vlm4wam-zero2-config-test`.

---

### Task 3: Replace manual DDP in the planner trainer

**Files:**
- Modify: `tests/test_lingbot_zero2_runtime.py`
- Modify: `tests/test_lingbot_dino_depth_contract.py`
- Modify: `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`

**Interfaces:**
- Consumes: all Task 1 runtime functions and existing `build_optimizer`, `build_scheduler`, `save_checkpoint`, dataset, teacher, and model APIs.
- Produces: an Accelerate-prepared planner/optimizer/DataLoader and optimizer-step bookkeeping that is correct for ZeRO-2 and ordinary Accelerate runs.

- [ ] **Step 1: Add failing trainer integration assertions**

Append this test to `tests/test_lingbot_zero2_runtime.py`:

```python
TRAINER = ROOT / "scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py"


def test_trainer_uses_accelerate_runtime_without_manual_ddp():
    source = TRAINER.read_text(encoding="utf-8")
    assert "accelerator.prepare(wrapper, optim, loader)" in source
    assert "accelerator.backward(out[\"loss\"])" in source
    assert "checkpoint_module(accelerator, wrapper)" in source
    assert "DistributedDataParallel" not in source
    assert "DistributedSampler" not in source
    assert ".backward()" not in source
```

Add this parser assertion beside the existing gradient-checkpointing tests in
`tests/test_lingbot_dino_depth_contract.py`:

```python
def test_expected_global_batch_cli_defaults_to_unconstrained_and_accepts_128(monkeypatch):
    module = load_trainer_module()
    monkeypatch.setattr(sys, "argv", _fastwam_parser_argv())
    assert module.parse_args().expected_global_batch == 0

    monkeypatch.setattr(
        sys,
        "argv",
        _fastwam_parser_argv("--expected-global-batch", "128"),
    )
    assert module.parse_args().expected_global_batch == 128
```

Remove the obsolete `ddp_info` monkeypatch from
`test_main_preflights_fastwam_before_loading_qwen`; preflight must still fail
before Accelerate or Qwen initialization.

- [ ] **Step 2: Run the focused trainer tests and verify RED**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py::test_trainer_uses_accelerate_runtime_without_manual_ddp tests/test_lingbot_dino_depth_contract.py::test_expected_global_batch_cli_defaults_to_unconstrained_and_accepts_128 tests/test_lingbot_dino_depth_contract.py::test_main_preflights_fastwam_before_loading_qwen
```

Expected: the integration and parser tests fail against the manual-DDP trainer; the preflight-order test continues to pass.

- [ ] **Step 3: Replace imports, CLI, and distributed initialization**

In the trainer, remove imports of `torch.distributed`, `DistributedDataParallel`,
and `DistributedSampler`. Import these symbols after adding
`lingbot_dino_4b` to `sys.path`:

```python
from distributed_runtime import (  # noqa: E402
    accumulation_context,
    build_accelerator,
    checkpoint_module,
    is_deepspeed,
    is_optimizer_update,
    validate_runtime_contract,
)
```

Add this CLI argument after `--grad-accum` and remove
`--ddp-find-unused-parameters`:

```python
parser.add_argument("--expected-global-batch", type=int, default=0)
```

Delete `ddp_info`. Preserve `is_main(rank)` because the checkpoint helper's
existing unit-test API still accepts an explicit rank.

Change the checkpoint helper annotation and unwrap line from DDP-specific code
to:

```python
def save_checkpoint(
    output_dir: Path,
    step: int,
    wrapper: PlannerWrapper,
    processor: Any,
    args: argparse.Namespace,
    rank: int,
) -> None:
    if not is_main(rank):
        return
    module = wrapper
```

After argument validation and FastWAM preflight, replace manual DDP setup with:

```python
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
accelerator = build_accelerator(grad_accum=args.grad_accum, dtype=args.dtype)
runtime_contract = validate_runtime_contract(
    accelerator,
    per_device_batch_size=args.batch_size,
    grad_accum=args.grad_accum,
    expected_global_batch=args.expected_global_batch,
)
rank = int(accelerator.process_index)
world = int(accelerator.num_processes)
device = accelerator.device
if accelerator.is_main_process:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "distributed_type": runtime_contract.distributed_type,
                "world_size": runtime_contract.world_size,
                "batch_size_per_gpu": runtime_contract.per_device_batch_size,
                "gradient_accumulation_steps": runtime_contract.gradient_accumulation_steps,
                "global_batch_size": runtime_contract.global_batch_size,
                "zero_stage": runtime_contract.zero_stage,
                "dtype": args.dtype,
                "gradient_checkpointing": bool(args.gradient_checkpointing),
            }
        ),
        flush=True,
    )
accelerator.wait_for_everyone()
random.seed(args.seed + rank)
torch.manual_seed(args.seed + rank)
```

- [ ] **Step 4: Prepare the model, optimizer, and DataLoader exactly once**

Remove the `DDP(...)` block. Build the DataLoader without a manual sampler:

```python
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    collate_fn=Collator(
        processor=processor,
        plan_sequence=plan_sequence,
    ),
    pin_memory=True,
)
optim = build_optimizer(wrapper, args)
scheduler = build_scheduler(optim, args)
wrapper, optim, loader = accelerator.prepare(wrapper, optim, loader)
```

Replace every training-time
`wrapper.module if isinstance(wrapper, DDP) else wrapper` with
`accelerator.unwrap_model(wrapper)`. Gate parameter logging, tqdm, and W&B on
`accelerator.is_main_process`. Remove the `sampler.set_epoch` branch from the
outer loop; retain `dataset.set_epoch(step)`.

- [ ] **Step 5: Replace the backward/update loop with a ZeRO-safe loop**

Initialize the loop with:

```python
step = 0
micro_step = 0
running_loss = 0.0
deepspeed_enabled = is_deepspeed(accelerator)
if not deepspeed_enabled:
    optim.zero_grad(set_to_none=True)
pbar = tqdm(
    total=args.max_steps,
    disable=not accelerator.is_local_main_process,
    desc="qwen3vl planner",
)
```

Replace the current direct backward and `accum` branch with:

```python
with accumulation_context(accelerator, wrapper):
    out = wrapper(**batch)
    accelerator.backward(out["loss"])
    if not deepspeed_enabled:
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(
                (parameter for parameter in wrapper.parameters() if parameter.requires_grad),
                1.0,
            )
        optim.step()
        optim.zero_grad(set_to_none=True)
running_loss += float(out["loss"].detach())
micro_step += 1
if is_optimizer_update(accelerator, micro_step, args.grad_accum):
    if scheduler is not None:
        scheduler.step()
    step += 1
    if accelerator.is_main_process:
        pbar.update(1)
        if step % args.log_steps == 0:
            average_loss = running_loss / max(args.log_steps * args.grad_accum, 1)
            log_entry = {
                "step": step,
                "loss": average_loss,
                "lr": scheduler.get_last_lr()[0] if scheduler is not None else args.lr,
            }
            log_entry.update(
                {key: float(value) for key, value in out.items() if key != "loss"}
            )
            print(json.dumps(log_entry), flush=True)
            if wandb_run is not None:
                wandb_run.log(log_entry, step=step)
    if step % args.log_steps == 0:
        running_loss = 0.0
    if step % args.save_steps == 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                args.output_dir,
                step,
                checkpoint_module(accelerator, wrapper),
                processor,
                args,
                rank=0,
            )
        accelerator.wait_for_everyone()
    if step >= args.max_steps:
        break
```

At finalization, use the same synchronized unwrapped save, finish W&B only on
the main process, and call `accelerator.end_training()` instead of manually
destroying a process group.

- [ ] **Step 6: Verify the trainer refactor**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py tests/test_lingbot_dino_depth_contract.py
python -m py_compile scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py
```

Expected: both test files pass and Python compilation produces no output.

---

### Task 4: Make one generic launcher and two machine profiles

**Files:**
- Modify: `tests/test_lingbot_zero2_runtime.py`
- Modify: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh`
- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh`
- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch`

**Interfaces:**
- Generic launcher consumes machine paths and hyperparameters through environment variables.
- Profiles produce complete machine-specific environments and call only `train_lingbot_dino_4b.sh`.

- [ ] **Step 1: Add a failing canonical-launcher contract test**

Append:

```python
GENERIC_LAUNCHER = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
    / "train_lingbot_dino_4b.sh"
)
POD_LAUNCHER = GENERIC_LAUNCHER.with_name("train_lingbot_fastwam_pod.sh")
HPC_LAUNCHER = GENERIC_LAUNCHER.with_name("train_lingbot_fastwam_hpc3.sbatch")


def test_only_canonical_launchers_are_referenced():
    generic = GENERIC_LAUNCHER.read_text(encoding="utf-8")
    pod = POD_LAUNCHER.read_text(encoding="utf-8")
    hpc = HPC_LAUNCHER.read_text(encoding="utf-8")
    assert "make_zero2_config.py" in generic
    assert "--expected-global-batch" in generic
    assert "NUM_TASK_TOKENS=${NUM_TASK_TOKENS:-64}" in generic
    assert "BATCH_SIZE=${BATCH_SIZE:-8}" in generic
    assert "GRAD_ACCUM=${GRAD_ACCUM:-2}" in generic
    assert "train_lingbot_dino_4b.sh" in pod
    assert "train_lingbot_dino_4b.sh" in hpc
    assert "train_lingbot_current_future_fastwam_k1.sh" not in pod + hpc
```

Also update the existing `_capture_base_launcher_args` test environment in
`tests/test_lingbot_dino_depth_contract.py` so its fake Python process exercises
the direct single-process path:

```python
"USE_DEEPSPEED": "0",
"BATCH_SIZE": "1",
"GRAD_ACCUM": "1",
"EXPECTED_GLOBAL_BATCH": "1",
```

- [ ] **Step 2: Run the launcher test and verify RED**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py::test_only_canonical_launchers_are_referenced
```

Expected: FAIL because the two canonical profiles do not exist and the generic launcher still uses torchrun defaults.

- [ ] **Step 3: Update the generic launch defaults and arguments**

Set these generic defaults in `train_lingbot_dino_4b.sh`:

```bash
NUM_GPUS=${NUM_GPUS:-8}
USE_DEEPSPEED=${USE_DEEPSPEED:-1}
USE_DEPTH=${USE_DEPTH:-1}
USE_CURRENT_ALIGNMENT=${USE_CURRENT_ALIGNMENT:-1}
INDEPENDENT_MODALITY_TASK_TOKENS=${INDEPENDENT_MODALITY_TASK_TOKENS:-1}
NUM_TASK_TOKENS=${NUM_TASK_TOKENS:-64}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-9}
NUM_KEYFRAMES=${NUM_KEYFRAMES:-1}
GRID_SIZE=${GRID_SIZE:-16}
KEYFRAME_SCHEME=${KEYFRAME_SCHEME:-even_future}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
EXPECTED_GLOBAL_BATCH=${EXPECTED_GLOBAL_BATCH:-128}
MAX_STEPS=${MAX_STEPS:-12000}
LR=${LR:-3e-5}
HEAD_LR=${HEAD_LR:-3e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
```

Add this trainer argument:

```bash
--expected-global-batch "$EXPECTED_GLOBAL_BATCH"
```

Remove `--ddp-find-unused-parameters`. Replace the torchrun tail with:

```bash
CONFIG_DIR="$OUTPUT_DIR/runtime_config"
mkdir -p "$CONFIG_DIR"
if [[ "$USE_DEEPSPEED" == "1" ]]; then
  ACCELERATE_CONFIG=$(
    "$PY" "$HERE/make_zero2_config.py" \
      --grad-accum "$GRAD_ACCUM" \
      --num-processes "$NUM_GPUS" \
      --output-dir "$CONFIG_DIR"
  )
  exec "$PY" -m accelerate.commands.launch \
    --config_file "$ACCELERATE_CONFIG" \
    "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
fi
exec "$PY" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
```

- [ ] **Step 4: Add the POD profile**

Create `train_lingbot_fastwam_pod.sh` as a thin profile that validates paths,
exports offline/cache settings, and invokes the generic launcher:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/nas/junjie}
REPO_ROOT=${REPO_ROOT:-$ROOT/code/VLM4WAM_k1_zero2_20260713}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/LIBERO-fastwam}
WEIGHTS=${WEIGHTS:-$ROOT/weights}
PY=${PY:-/opt/conda/envs/vlm4wam/bin/python}
RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
if [[ "$RUN_KIND" == "smoke" ]]; then
  MAX_STEPS=${MAX_STEPS:-2}
  SAVE_STEPS=${SAVE_STEPS:-2}
else
  MAX_STEPS=${MAX_STEPS:-12000}
  SAVE_STEPS=${SAVE_STEPS:-1000}
fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl4b_lingbot_independent_q64_zero2_k1_b${BATCH_SIZE}a${GRAD_ACCUM}}

for path in \
  "$PY" \
  "$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  "$WEIGHTS/lingbot_align_heads_warmstart/model.safetensors.index.json" \
  "$WEIGHTS/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
  "$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  "$WEIGHTS/moge-2-vitb-normal/model.pt" \
  "$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" \
  "$ROOT/data/libero_qwen" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" \
  "$DATA_ROOT/libero_object_no_noops_lerobot" \
  "$DATA_ROOT/libero_goal_no_noops_lerobot" \
  "$DATA_ROOT/libero_10_no_noops_lerobot"; do
  [[ -e "$path" ]] || { echo "ERROR: missing required path: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs" "$ROOT/cache/triton" "$ROOT/cache/inductor"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PLANNER_WANDB=${PLANNER_WANDB:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "$REPO_ROOT"
exec env \
  NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
  EXPECTED_GLOBAL_BATCH=128 MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  FULL_FINETUNE=1 NUM_WORKERS=4 LR=3e-5 HEAD_LR=3e-4 WARMUP_STEPS=1000 \
  PY="$PY" WEIGHTS="$WEIGHTS" \
  MODEL_PATH="$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  LINGBOT_6B="$WEIGHTS/lingbot-vla-v2-6b" \
  HEAD_WARMSTART_CKPT="$WEIGHTS/lingbot_align_heads_warmstart" \
  DEPTH_MOGE_PATH="$WEIGHTS/moge-2-vitb-normal/model.pt" \
  DEPTH_MORGBD_PATH="$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  LINGBOT_SRC_ROOT="$ROOT/code/lingbot-vla-v2" \
  UTILS3D_MOGE_PATH="$ROOT/py_deps/utils3d_moge" \
  FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  FASTWAM_DATASET_DIRS="$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot" \
  FASTWAM_TEXT_EMBEDDING_CACHE_DIR="$ROOT/data/libero_qwen" \
  FASTWAM_PRETRAINED_NORM_STATS="$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
```

- [ ] **Step 5: Add the HPC3 profile**

Create `train_lingbot_fastwam_hpc3.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH -J vlmp_zero2_q64
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

WORKSPACE=${WORKSPACE:-/data/user/jhe724/workspace}
REPO_ROOT=${REPO_ROOT:-$WORKSPACE/VLM4WAM_lingbot_zero2_20260713}
DATA_ROOT=${DATA_ROOT:-$WORKSPACE/datasets/LIBERO-fastwam}
TEXT_CACHE=${TEXT_CACHE:-$WORKSPACE/datasets/libero_qwen}
NORM_STATS=${NORM_STATS:-$WORKSPACE/datasets/LIBERO-fastwam_meta/dataset_stats.json}
WEIGHTS=${WEIGHTS:-$WORKSPACE/weights}
PY=${PY:-$HOME/.conda/envs/starVLA/bin/python}
RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=8
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
if [[ "$RUN_KIND" == "smoke" ]]; then
  MAX_STEPS=${MAX_STEPS:-2}
  SAVE_STEPS=${SAVE_STEPS:-2}
else
  MAX_STEPS=${MAX_STEPS:-12000}
  SAVE_STEPS=${SAVE_STEPS:-1000}
fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl4b_lingbot_independent_q64_zero2_k1_b${BATCH_SIZE}a${GRAD_ACCUM}_${SLURM_JOB_ID}}

for path in \
  "$PY" \
  "$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  "$WEIGHTS/lingbot_align_heads_warmstart/model.safetensors.index.json" \
  "$WEIGHTS/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
  "$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  "$WEIGHTS/moge-2-vitb-normal/model.pt" \
  "$NORM_STATS" \
  "$TEXT_CACHE" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" \
  "$DATA_ROOT/libero_object_no_noops_lerobot" \
  "$DATA_ROOT/libero_goal_no_noops_lerobot" \
  "$DATA_ROOT/libero_10_no_noops_lerobot"; do
  [[ -e "$path" ]] || { echo "ERROR: missing required path: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs" "$WORKSPACE/.cache/triton" "$WORKSPACE/.cache/inductor"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PLANNER_WANDB=${PLANNER_WANDB:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$WORKSPACE/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$WORKSPACE/.cache/inductor
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "$REPO_ROOT"
exec env \
  NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
  EXPECTED_GLOBAL_BATCH=128 MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  FULL_FINETUNE=1 NUM_WORKERS=4 LR=3e-5 HEAD_LR=3e-4 WARMUP_STEPS=1000 \
  PY="$PY" WEIGHTS="$WEIGHTS" \
  MODEL_PATH="$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  LINGBOT_6B="$WEIGHTS/lingbot-vla-v2-6b" \
  HEAD_WARMSTART_CKPT="$WEIGHTS/lingbot_align_heads_warmstart" \
  DEPTH_MOGE_PATH="$WEIGHTS/moge-2-vitb-normal/model.pt" \
  DEPTH_MORGBD_PATH="$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  LINGBOT_SRC_ROOT="$WORKSPACE/lingbot-vla-v2" \
  UTILS3D_MOGE_PATH="$WORKSPACE/py_deps/utils3d_moge" \
  FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  FASTWAM_DATASET_DIRS="$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot" \
  FASTWAM_TEXT_EMBEDDING_CACHE_DIR="$TEXT_CACHE" \
  FASTWAM_PRETRAINED_NORM_STATS="$NORM_STATS" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
```

- [ ] **Step 6: Verify launcher tests and shell syntax**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py::test_only_canonical_launchers_are_referenced
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch
```

Expected: the test passes and all three shell checks produce no output.

---

### Task 5: Remove superseded wrappers and launcher-only tests

**Files:**
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1.sh`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1_pod30274.sh`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1_hpc3.sbatch`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4.sh`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4_hpc3.sbatch`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_pod30274.sh`
- Delete: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_hpc3.sbatch`
- Modify: `tests/test_lingbot_k1_current_future.py`
- Modify: `tests/test_lingbot_dino_depth_contract.py`
- Modify: `scripts/qwen3_vl_semantic_planner/README.md`

**Interfaces:**
- Preserves all behavioral query/teacher/loss/export/provider tests.
- Removes only tests whose sole observable is literal content in a superseded wrapper.

- [ ] **Step 1: Remove obsolete launcher-only assertions**

Delete these test functions:

```text
tests/test_lingbot_k1_current_future.py::test_independent_query_pod_launcher_pins_fair_ablation_contract
tests/test_lingbot_k1_current_future.py::test_independent_query_hpc3_launcher_pins_64_tokens_per_feature
tests/test_lingbot_dino_depth_contract.py::test_fastwam_launcher_pins_nine_frame_dual_branch_contract
tests/test_lingbot_dino_depth_contract.py::test_hpc3_launcher_defaults_to_recommended_12k_budget
```

Keep the generic-launcher cache/path tests because the generic launcher remains
production code. Keep all depth-probe and planner-evaluation files and tests.

- [ ] **Step 2: Delete the seven superseded wrappers**

Use an `apply_patch` delete operation for each exact file listed under Task 5.

- [ ] **Step 3: Update the README to the canonical contract**

Document:

```text
Qwen3-VL 4B LingBot FastWAM current configuration:
- frames: current 0 and future 8 from a 9-frame sample
- VLM queries: 4 independent groups × 64 = 256 tokens
- outputs: current/future DINO and current/future depth, each 256 × 1024
- distributed runtime: Accelerate + DeepSpeed ZeRO-2
- preferred batch: 8 GPUs × 8/GPU × accumulation 2 = global 128
- generic launcher: lingbot_dino_4b/train_lingbot_dino_4b.sh
- POD profile: lingbot_dino_4b/train_lingbot_fastwam_pod.sh
- HPC3 profile: lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch
```

- [ ] **Step 4: Prove no live references remain**

Run:

```bash
rg -n "train_lingbot_(current_future_fastwam_k1|dino_depth_fastwam_k4|independent_queries_fastwam_k1)" scripts tests docs -S
```

Expected: matches may remain only in historical design/plan documents; no match may remain in executable scripts, tests, or the current README.

---

### Task 6: Run the full local regression and static verification

**Files:**
- Verify all implementation and test files from Tasks 1–5.

**Interfaces:**
- Produces a locally verified candidate for the remote two-step smoke run.

- [ ] **Step 1: Run focused planner and FastWAM tests**

Run:

```bash
pytest -q \
  tests/test_lingbot_zero2_runtime.py \
  tests/test_lingbot_k1_current_future.py \
  tests/test_lingbot_dino_depth_contract.py \
  tests/test_dino_depth_plan_provider.py \
  tests/test_fastwam_online_semantic_planner.py \
  tests/test_fastwam_semantic_timing_routing.py \
  tests/test_fastwam_cosmos_semantic_plan.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Compile Python and validate shell files**

Run:

```bash
python -m py_compile \
  scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch
git diff --check
```

Expected: every command exits zero and produces no error output.

- [ ] **Step 3: Review the scoped diff without staging user changes**

Run:

```bash
git status --short
git diff -- scripts/qwen3_vl_semantic_planner tests scripts/qwen3_vl_semantic_planner/README.md
```

Expected: the diff contains the approved runtime, launcher, test cleanup, and
README changes and preserves all unrelated dirty-worktree files.

---

### Task 7: Deploy an isolated POD smoke run and validate the checkpoint

**Files:**
- Remote copy: `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713`
- Remote output: `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/smoke_zero2_b8a2`

**Interfaces:**
- Consumes the verified local implementation and existing POD weights/data.
- Produces a two-step ZeRO-2 checkpoint compatible with the FastWAM provider.

- [ ] **Step 1: Invoke the run-experiment skill before remote mutation**

Read and follow `/home/LFT-W02/.codex/skills/run-experiment/SKILL.md`. Confirm
the active long-running PID is not targeted by any command.

- [ ] **Step 2: Create an isolated remote code copy and sync only scoped files**

Run from the local repository root:

```bash
ssh -p 30282 root@182.242.159.145 \
  'test ! -e /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713 && cp -a /root/nas/junjie/code/VLM4WAM_k1_fastwam_20260712 /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713'
rsync -avR -e 'ssh -p 30282' \
  ./scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py \
  ./scripts/qwen3_vl_semantic_planner/README.md \
  ./scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py \
  ./scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py \
  ./scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh \
  ./scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh \
  ./scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch \
  ./tests/test_lingbot_zero2_runtime.py \
  ./tests/test_lingbot_dino_depth_contract.py \
  ./tests/test_lingbot_k1_current_future.py \
  root@182.242.159.145:/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/
ssh -p 30282 root@182.242.159.145 \
  'rm -f /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1.sh /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1_pod30274.sh /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1_hpc3.sbatch /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4.sh /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4_hpc3.sbatch /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_pod30274.sh /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_hpc3.sbatch'
```

These commands do not sync outputs, logs, caches, `.git`, or unrelated dirty
files.

- [ ] **Step 3: Launch the preferred b8/a2 smoke**

From the isolated remote repository, run:

```bash
RUN_KIND=smoke \
REPO_ROOT=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713 \
OUTPUT_DIR=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/smoke_zero2_b8a2 \
MAX_STEPS=2 SAVE_STEPS=2 BATCH_SIZE=8 GRAD_ACCUM=2 \
bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh
```

Capture stdout/stderr in a new log and record only the new wrapper/rank PIDs.

- [ ] **Step 4: Monitor runtime, memory, and completion**

Verify the log reports:

```text
distributed_type=DEEPSPEED
world_size=8
batch_size_per_gpu=8
gradient_accumulation_steps=2
global_batch_size=128
zero_stage=2
gradient_checkpointing=false
```

Also verify all eight ranks are alive during training, GPU memory stays below
the device limit, and the log contains no OOM, traceback, NCCL error, NaN, or
accumulation mismatch.

- [ ] **Step 5: Use the documented memory fallback only if b8/a2 OOMs**

If and only if the smoke fails with CUDA OOM, stop only the smoke PIDs and
repeat with:

```bash
BATCH_SIZE=4 GRAD_ACCUM=4 OUTPUT_DIR=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/smoke_zero2_b4a4
```

Do not change global batch, query geometry, loss weights, checkpointing, or the
active pre-existing training process.

- [ ] **Step 6: Validate the step-2 export**

Verify `step_000002` contains the model, processor, all four head files,
`plan_token_embedding.pt`, and `planner_meta.json`. Parse metadata and assert:

```text
num_keyframes = 1
num_task_tokens = 64
latent_len = 256
total_unique_latent_per_keyframe = 256
target_tokens = 256
independent_modality_task_tokens = true
keyframe_offsets = [8]
```

Load the metadata through `dino_depth_plan_provider.validate_planner_metadata`
to prove the production consumer accepts the export. Run:

```bash
ssh -p 30282 root@182.242.159.145 \
  'cd /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713 && /opt/conda/envs/vlm4wam/bin/python -c "import json,sys; from pathlib import Path; root=Path(\"outputs/smoke_zero2_b8a2/step_000002\"); required=(\"qwen3vl_lora_or_model\",\"processor\",\"plan_head.pt\",\"depth_head.pt\",\"current_plan_head.pt\",\"current_depth_head.pt\",\"plan_token_embedding.pt\",\"planner_meta.json\"); missing=[name for name in required if not (root/name).exists()]; assert not missing, missing; meta=json.loads((root/\"planner_meta.json\").read_text()); assert meta[\"num_keyframes\"] == 1; assert meta[\"num_task_tokens\"] == 64; assert meta[\"latent_len\"] == 256; assert meta[\"total_unique_latent_per_keyframe\"] == 256; assert meta[\"target_tokens\"] == 256; assert meta[\"independent_modality_task_tokens\"] is True; assert meta[\"keyframe_offsets\"] == [8]; sys.path.insert(0, str(Path(\"scripts/qwen3_vl_semantic_planner/lingbot_dino_4b\").resolve())); from dino_depth_plan_provider import validate_planner_metadata; contract=validate_planner_metadata(meta); assert contract.num_task_tokens == 64; print(\"step-2 export valid\")"'
```

Expected: the remote command prints `step-2 export valid`.
