from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
    / "distributed_runtime.py"
)
CONFIG_GENERATOR = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
    / "make_zero2_config.py"
)
TRAINER = ROOT / "scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py"
GENERIC_LAUNCHER = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
    / "train_lingbot_dino_4b.sh"
)
POD_LAUNCHER = GENERIC_LAUNCHER.with_name("train_lingbot_fastwam_pod.sh")
HPC_LAUNCHER = GENERIC_LAUNCHER.with_name("train_lingbot_fastwam_hpc3.sbatch")


def load_runtime():
    spec = importlib.util.spec_from_file_location("lingbot_distributed_runtime", RUNTIME)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_config_generator():
    spec = importlib.util.spec_from_file_location("lingbot_zero2_config", CONFIG_GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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


def test_trainer_uses_accelerate_runtime_without_manual_ddp():
    source = TRAINER.read_text(encoding="utf-8")
    assert "accelerator.prepare(wrapper, optim, loader)" in source
    assert "accelerator.backward(out[\"loss\"])" in source
    assert "checkpoint_module(accelerator, wrapper)" in source
    assert "DistributedDataParallel" not in source
    assert "DistributedSampler" not in source
    assert ".backward()" not in source


def test_trainer_moves_prepared_siglip_keyframes_to_cpu_before_numpy():
    source = TRAINER.read_text(encoding="utf-8")
    assert "keyframes[i, j].detach().cpu().numpy()" in source
    assert "keyframes[i, j].numpy()" not in source


def test_only_canonical_launchers_are_referenced():
    generic = GENERIC_LAUNCHER.read_text(encoding="utf-8")
    pod = POD_LAUNCHER.read_text(encoding="utf-8")
    hpc = HPC_LAUNCHER.read_text(encoding="utf-8")
    assert "set -euo pipefail" in generic
    assert "make_zero2_config.py" in generic
    assert "--expected-global-batch" in generic
    assert "NUM_TASK_TOKENS=${NUM_TASK_TOKENS:-64}" in generic
    assert "BATCH_SIZE=${BATCH_SIZE:-8}" in generic
    assert "GRAD_ACCUM=${GRAD_ACCUM:-2}" in generic
    assert "train_lingbot_dino_4b.sh" in pod
    assert "train_lingbot_dino_4b.sh" in hpc
    assert "train_lingbot_current_future_fastwam_k1.sh" not in pod + hpc
