"""Pinned-memory CUDA prefetch for Baton Stage-1 batches."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Iterable, Iterator, Mapping

import torch


class _CudaStreams:
    def create(self, device: torch.device) -> torch.cuda.Stream:
        return torch.cuda.Stream(device=device)

    def use(self, stream: torch.cuda.Stream):
        return torch.cuda.stream(stream)

    def current_stream(self, device: torch.device) -> torch.cuda.Stream:
        return torch.cuda.current_stream(device=device)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _to_device(item, device)) for key, item in value.items()
        )
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_to_device(item, device) for item in value))
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    move = getattr(value, "to", None)
    if callable(move):
        return move(device, non_blocking=True)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _to_device(getattr(value, field.name), device)
                for field in fields(value)
            }
        )
    return value


def _record_stream(value: Any, stream: Any) -> None:
    record = getattr(value, "record_stream", None)
    if callable(record):
        record(stream)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record_stream(item, stream)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _record_stream(getattr(value, field.name), stream)


class DevicePrefetchLoader:
    """Overlap the next pinned host batch transfer with GPU computation."""

    def __init__(
        self,
        base_dataloader: Iterable[Any],
        device: torch.device,
        *,
        stream_factory: Any | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("device prefetch requires a CUDA device")
        self.base_dataloader = base_dataloader
        self.device = device
        self._streams = _CudaStreams() if stream_factory is None else stream_factory

    def __len__(self) -> int:
        return len(self.base_dataloader)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataloader, name)

    def __iter__(self) -> Iterator[Any]:
        source = iter(self.base_dataloader)
        transfer_stream = self._streams.create(self.device)
        sentinel = object()
        next_batch: Any = sentinel

        def preload() -> None:
            nonlocal next_batch
            try:
                host_batch = next(source)
            except StopIteration:
                next_batch = sentinel
                return
            with self._streams.use(transfer_stream):
                next_batch = _to_device(host_batch, self.device)

        preload()
        while next_batch is not sentinel:
            current_stream = self._streams.current_stream(self.device)
            current_stream.wait_stream(transfer_stream)
            batch = next_batch
            _record_stream(batch, current_stream)
            preload()
            yield batch


def enable_device_prefetch(
    loader: Iterable[Any],
    device: torch.device | str,
    *,
    stream_factory: Any | None = None,
) -> Iterable[Any]:
    """Wrap CUDA loaders; leave CPU loaders unchanged for tests and smoke runs."""

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return loader
    return DevicePrefetchLoader(
        loader,
        resolved,
        stream_factory=stream_factory,
    )
