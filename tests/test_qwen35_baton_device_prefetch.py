from __future__ import annotations

from contextlib import nullcontext

import torch


class _PinAwareBatch:
    def __init__(self) -> None:
        self.non_blocking: bool | None = None
        self.recorded_stream = None

    def to(self, device: torch.device, *, non_blocking: bool):
        assert device == torch.device("cuda:0")
        self.non_blocking = non_blocking
        return self

    def record_stream(self, stream) -> None:
        self.recorded_stream = stream


class _FakeStream:
    def __init__(self) -> None:
        self.waited_for = None

    def wait_stream(self, stream) -> None:
        self.waited_for = stream


class _FakeStreamFactory:
    def __init__(self) -> None:
        self.transfer = _FakeStream()
        self.current = _FakeStream()

    def create(self, device: torch.device):
        assert device == torch.device("cuda:0")
        return self.transfer

    def use(self, stream):
        assert stream is self.transfer
        return nullcontext()

    def current_stream(self, device: torch.device):
        assert device == torch.device("cuda:0")
        return self.current


def test_prefetch_uses_nonblocking_transfer_and_records_stream() -> None:
    from qwen35_baton.device_prefetch import enable_device_prefetch

    batch = _PinAwareBatch()
    streams = _FakeStreamFactory()
    loader = enable_device_prefetch(
        [batch],
        torch.device("cuda:0"),
        stream_factory=streams,
    )

    received = next(iter(loader))

    assert received is batch
    assert batch.non_blocking is True
    assert batch.recorded_stream is streams.current
    assert streams.current.waited_for is streams.transfer
