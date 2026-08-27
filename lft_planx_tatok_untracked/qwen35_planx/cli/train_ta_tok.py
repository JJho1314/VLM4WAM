#!/usr/bin/env python3
"""Train the domain TA-Tok on both LIBERO camera streams."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    per_device_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    expected_global_batch_size: int

    def __post_init__(self) -> None:
        for name in (
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "world_size",
            "expected_global_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.global_batch_size != self.expected_global_batch_size:
            raise ValueError(
                f"global batch is {self.global_batch_size}, expected "
                f"{self.expected_global_batch_size}"
            )

    @property
    def global_batch_size(self) -> int:
        return (
            self.per_device_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
        )
