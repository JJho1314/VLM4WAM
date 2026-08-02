"""Low-synchronization telemetry helpers for Baton Stage-1 training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class Stage1MetricAccumulator:
    """Accumulate detached scalar metrics on device until a logging boundary."""

    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}

    def add_scalar(
        self,
        name: str,
        value: torch.Tensor | float,
        *,
        device: torch.device | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("metric name must be a nonempty string")
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("metric values must be scalar")
            tensor = value.detach().reshape(())
        else:
            tensor = torch.tensor(float(value), device=device)
        tensor = tensor.to(dtype=torch.float64)
        previous = self.sums.get(name)
        self.sums[name] = tensor if previous is None else previous + tensor

    def add_loss(
        self,
        total: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        camera_names: tuple[str, ...],
        mse: torch.Tensor | None = None,
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError("prediction and target must share [B,C,F,P,D] shape")
        if len(camera_names) != prediction.shape[1]:
            raise ValueError("camera_names must match the prediction camera axis")
        self.add_scalar("loss/total", total)
        self.add_scalar("loss/mse", total if mse is None else mse)
        per_camera_frame = (
            prediction.detach().float() - target.detach().float()
        ).square().mean(dim=(0, 3, 4))
        for camera_index, camera_name in enumerate(camera_names):
            for frame_index in range(prediction.shape[2]):
                self.add_scalar(
                    f"mse/{camera_name}/frame_{frame_index}",
                    per_camera_frame[camera_index, frame_index],
                )

    def flush(self, accelerator: Any, *, divisor: int) -> dict[str, float]:
        if type(divisor) is not int or divisor <= 0:
            raise ValueError("metric divisor must be a positive integer")
        if not self.sums:
            return {}
        names = tuple(sorted(self.sums))
        values = torch.stack(tuple(self.sums[name] for name in names)) / divisor
        values = accelerator.reduce(values, reduction="mean")
        result = {
            name: float(value)
            for name, value in zip(names, values.detach().cpu().tolist(), strict=True)
        }
        self.sums.clear()
        return result


class CudaEventTimer:
    """Record named CUDA intervals and synchronize only when resolving them."""

    def __init__(
        self,
        *,
        enabled: bool,
        event_factory: Callable[[], Any] | None = None,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._event_factory = event_factory or (
            lambda: torch.cuda.Event(enable_timing=True)
        )
        self._synchronize = synchronize or torch.cuda.synchronize
        self._active: dict[str, Any] = {}
        self._intervals: dict[str, list[tuple[Any, Any]]] = {}

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        if name in self._active:
            raise RuntimeError(f"timer interval is already active: {name}")
        event = self._event_factory()
        event.record()
        self._active[name] = event

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        try:
            start = self._active.pop(name)
        except KeyError as error:
            raise RuntimeError(f"timer interval is not active: {name}") from error
        stop = self._event_factory()
        stop.record()
        self._intervals.setdefault(name, []).append((start, stop))

    def resolve(self) -> dict[str, float]:
        if not self.enabled or not self._intervals:
            return {}
        if self._active:
            raise RuntimeError("cannot resolve active timer intervals")
        self._synchronize()
        result = {
            name: sum(start.elapsed_time(stop) for start, stop in intervals) / 1000.0
            for name, intervals in self._intervals.items()
        }
        self._intervals.clear()
        return result
