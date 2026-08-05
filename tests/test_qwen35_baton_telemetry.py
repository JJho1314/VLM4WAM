from __future__ import annotations

import torch

from qwen35_baton.training_telemetry import CudaEventTimer, Stage1MetricAccumulator


class _IdentityAccelerator:
    @staticmethod
    def reduce(value: torch.Tensor, *, reduction: str) -> torch.Tensor:
        assert reduction == "mean"
        return value


class _FakeEvent:
    def __init__(self, factory: "_FakeEventFactory") -> None:
        self.factory = factory
        self.record_index: int | None = None

    def record(self) -> None:
        self.record_index = self.factory.record_count
        self.factory.record_count += 1

    def elapsed_time(self, other: "_FakeEvent") -> float:
        assert self.record_index is not None
        assert other.record_index == self.record_index + 1
        return 125.0


class _FakeEventFactory:
    def __init__(self) -> None:
        self.record_count = 0
        self.synchronize_calls = 0

    def __call__(self) -> _FakeEvent:
        return _FakeEvent(self)

    def synchronize(self) -> None:
        self.synchronize_calls += 1


def test_metric_accumulator_keeps_values_as_tensors_until_flush() -> None:
    accumulator = Stage1MetricAccumulator()
    prediction = torch.zeros((1, 1, 4, 2, 3))
    target = torch.ones_like(prediction)

    accumulator.add_loss(
        torch.tensor(2.0),
        prediction,
        target,
        camera_names=("head",),
    )

    assert accumulator.sums
    assert all(isinstance(value, torch.Tensor) for value in accumulator.sums.values())
    metrics = accumulator.flush(_IdentityAccelerator(), divisor=1)
    assert metrics["loss/total"] == 2.0
    assert metrics["loss/mse"] == 2.0
    assert metrics["mse/head/frame_0"] == 1.0
    assert accumulator.sums == {}


def test_cuda_event_timer_only_synchronizes_when_resolved() -> None:
    factory = _FakeEventFactory()
    timer = CudaEventTimer(
        enabled=True,
        event_factory=factory,
        synchronize=factory.synchronize,
    )

    timer.start("qwen")
    timer.stop("qwen")

    assert factory.synchronize_calls == 0
    assert timer.resolve()["qwen"] == 0.125
    assert factory.synchronize_calls == 1
