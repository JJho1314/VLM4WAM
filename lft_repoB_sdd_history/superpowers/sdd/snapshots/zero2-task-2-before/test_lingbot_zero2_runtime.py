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
