"""Deterministic language-grounding diagnostics for Baton Stage-1 planners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from qwen35_baton.sequence import validate_source_indices


@dataclass(frozen=True)
class GroundingExample:
    """Auditable validation identity independent of tensor storage."""

    episode_id: str
    task: str
    instruction: str
    source_indices: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        for name in ("episode_id", "task", "instruction"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string")
        validate_source_indices(self.source_indices)


@dataclass(frozen=True)
class GroundingGateDecision:
    """Result and immutable thresholds for downstream GE-Act eligibility."""

    eligible: bool
    failures: tuple[str, ...]
    required_examples: int = 44
    minimum_correct_win_fraction: float = 0.60
    minimum_shuffle_mse_improvement: float = 0.05
    minimum_persistence_mse_improvement: float = 0.25
    accepted_norm_ratio: tuple[float, float] = (0.85, 1.15)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failures"] = list(self.failures)
        payload["accepted_norm_ratio"] = list(self.accepted_norm_ratio)
        return payload


def task_distinct_shuffle(
    examples: Sequence[GroundingExample],
    *,
    seed: int,
) -> tuple[int, ...]:
    """Return a deterministic bijection whose paired tasks always differ."""

    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise TypeError("examples must be a sequence")
    if not examples or any(not isinstance(value, GroundingExample) for value in examples):
        raise ValueError("examples must contain GroundingExample values")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if len({example.task for example in examples}) < 2:
        raise ValueError("task-distinct shuffle is impossible with one task")

    rng = random.Random(seed)
    candidates: list[list[int]] = []
    for example in examples:
        row = [
            index
            for index, candidate in enumerate(examples)
            if candidate.task != example.task
        ]
        rng.shuffle(row)
        candidates.append(row)
    source_for_target = [-1] * len(examples)

    def assign(source: int, seen: set[int]) -> bool:
        for target in candidates[source]:
            if target in seen:
                continue
            seen.add(target)
            previous = source_for_target[target]
            if previous < 0 or assign(previous, seen):
                source_for_target[target] = source
                return True
        return False

    source_order = list(range(len(examples)))
    rng.shuffle(source_order)
    for source in source_order:
        if not assign(source, set()):
            raise ValueError(
                "task-distinct shuffle has no complete cross-task assignment"
            )
    target_for_source = [-1] * len(examples)
    for target, source in enumerate(source_for_target):
        target_for_source[source] = target
    if any(value < 0 for value in target_for_source):
        raise AssertionError("task-distinct matching did not produce a bijection")
    return tuple(target_for_source)


def _validate_feature_tensor(name: str, value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 4
        or value.shape[0] <= 0
        or value.shape[1] != 4
        or value.shape[2] <= 0
        or value.shape[3] <= 0
        or not value.dtype.is_floating_point
    ):
        raise ValueError(f"{name} must be floating-point [N,4,P,D]")


def _sample_mse(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (value - target).square().mean(dim=(1, 2, 3))


def _sample_cosine(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(value.flatten(1), target.flatten(1), dim=1)


def _mean(value: torch.Tensor) -> float:
    return float(value.to(dtype=torch.float64).mean().item())


def _relative_improvement(candidate: float, baseline: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline) or baseline <= 0:
        return -1.0
    return (baseline - candidate) / baseline


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return -1.0
    return numerator / denominator


def compute_grounding_metrics(
    *,
    correct: torch.Tensor,
    shuffled: torch.Tensor,
    target: torch.Tensor,
    persistence: torch.Tensor,
) -> dict[str, Any]:
    """Compute deterministic aggregate, horizon, scale, and sample diagnostics."""

    tensors = {
        "correct": correct,
        "shuffled": shuffled,
        "target": target,
        "persistence": persistence,
    }
    for name, value in tensors.items():
        _validate_feature_tensor(name, value)
    if any(value.shape != target.shape for value in tensors.values()):
        raise ValueError("all grounding feature tensors must have identical shape")
    finite = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
    for value in tensors.values():
        finite &= torch.isfinite(value).flatten(1).all(dim=1)
    finite_indices = torch.nonzero(finite, as_tuple=False).flatten()
    if finite_indices.numel() == 0:
        raise ValueError("grounding validation has no finite examples")

    selected = {
        name: value.index_select(0, finite_indices).to(dtype=torch.float32)
        for name, value in tensors.items()
    }
    sample_mse = {
        name: _sample_mse(selected[name], selected["target"])
        for name in ("correct", "shuffled", "persistence")
    }
    sample_cosine = {
        name: _sample_cosine(selected[name], selected["target"])
        for name in ("correct", "shuffled", "persistence")
    }
    aggregate = {
        f"{name}_mse": _mean(sample_mse[name])
        for name in ("correct", "shuffled", "persistence")
    }
    aggregate.update(
        {
            f"{name}_cosine": _mean(sample_cosine[name])
            for name in ("correct", "shuffled", "persistence")
        }
    )

    per_horizon: list[dict[str, float | int]] = []
    for horizon in range(4):
        row: dict[str, float | int] = {"horizon": horizon}
        horizon_target = selected["target"][:, horizon].flatten(1)
        for name in ("correct", "shuffled", "persistence"):
            horizon_value = selected[name][:, horizon].flatten(1)
            row[f"{name}_mse"] = _mean(
                (horizon_value - horizon_target).square().mean(dim=1)
            )
            row[f"{name}_cosine"] = _mean(
                F.cosine_similarity(horizon_value, horizon_target, dim=1)
            )
        per_horizon.append(row)

    prediction_norm = _mean(torch.linalg.vector_norm(selected["correct"], dim=-1))
    target_norm = _mean(torch.linalg.vector_norm(selected["target"], dim=-1))
    prediction_spatial_std = _mean(
        torch.linalg.vector_norm(
            selected["correct"].std(dim=-2, unbiased=False), dim=-1
        )
    )
    target_spatial_std = _mean(
        torch.linalg.vector_norm(
            selected["target"].std(dim=-2, unbiased=False), dim=-1
        )
    )
    delta_prediction = (selected["correct"] - selected["persistence"]).flatten(2)
    delta_target = (selected["target"] - selected["persistence"]).flatten(2)
    future_delta_cosine = _mean(
        F.cosine_similarity(delta_prediction, delta_target, dim=2)
    )
    wins = sample_mse["correct"] < sample_mse["shuffled"]

    per_sample: list[dict[str, Any]] = []
    finite_cursor = 0
    for is_finite in finite.tolist():
        if not is_finite:
            per_sample.append(
                {
                    "finite": False,
                    "correct_mse": None,
                    "shuffled_mse": None,
                    "persistence_mse": None,
                }
            )
            continue
        per_sample.append(
            {
                "finite": True,
                "correct_mse": float(sample_mse["correct"][finite_cursor].item()),
                "shuffled_mse": float(sample_mse["shuffled"][finite_cursor].item()),
                "persistence_mse": float(
                    sample_mse["persistence"][finite_cursor].item()
                ),
            }
        )
        finite_cursor += 1

    correct_mse = float(aggregate["correct_mse"])
    shuffled_mse = float(aggregate["shuffled_mse"])
    persistence_mse = float(aggregate["persistence_mse"])
    return {
        "example_count": int(target.shape[0]),
        "finite_example_count": int(finite.sum().item()),
        "aggregate": aggregate,
        "per_horizon": per_horizon,
        "correct_win_count": int(wins.sum().item()),
        "correct_win_fraction": _mean(wins.to(dtype=torch.float32)),
        "shuffle_mse_improvement": _relative_improvement(
            correct_mse, shuffled_mse
        ),
        "persistence_mse_improvement": _relative_improvement(
            correct_mse, persistence_mse
        ),
        "prediction_target_norm_ratio": _safe_ratio(
            prediction_norm, target_norm
        ),
        "prediction_target_spatial_std_ratio": _safe_ratio(
            prediction_spatial_std, target_spatial_std
        ),
        "future_delta_cosine": future_delta_cosine,
        "per_sample": per_sample,
    }


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def evaluate_grounding_gates(
    metrics: Mapping[str, Any],
) -> GroundingGateDecision:
    """Evaluate all approved checkpoint-eligibility gates without side effects."""

    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    failures: list[str] = []
    example_count = _finite_metric(metrics, "example_count")
    finite_count = _finite_metric(metrics, "finite_example_count")
    if example_count != 44 or finite_count != 44:
        failures.append("finite_examples")
    checks = (
        ("correct_win_fraction", 0.60),
        ("shuffle_mse_improvement", 0.05),
        ("persistence_mse_improvement", 0.25),
    )
    for name, minimum in checks:
        value = _finite_metric(metrics, name)
        if value is None or value < minimum:
            failures.append(name)
    norm_ratio = _finite_metric(metrics, "prediction_target_norm_ratio")
    if norm_ratio is None or not 0.85 <= norm_ratio <= 1.15:
        failures.append("prediction_target_norm_ratio")
    return GroundingGateDecision(
        eligible=not failures,
        failures=tuple(failures),
    )


def build_grounding_artifact(
    *,
    step: int,
    examples: Sequence[GroundingExample],
    shuffled_indices: Sequence[int],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine metrics and task-distinct pairing into a JSON-safe artifact."""

    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    if len(examples) != len(shuffled_indices):
        raise ValueError("examples and shuffled_indices must have equal lengths")
    if sorted(shuffled_indices) != list(range(len(examples))):
        raise ValueError("shuffled_indices must be a complete permutation")
    per_sample = metrics.get("per_sample")
    if not isinstance(per_sample, list) or len(per_sample) != len(examples):
        raise ValueError("metrics per_sample must align with examples")
    samples: list[dict[str, Any]] = []
    for index, (example, shuffled_index) in enumerate(
        zip(examples, shuffled_indices)
    ):
        shuffled = examples[shuffled_index]
        if example.task == shuffled.task:
            raise ValueError("grounding artifact contains a same-task shuffle")
        samples.append(
            {
                "episode_id": example.episode_id,
                "task": example.task,
                "instruction": example.instruction,
                "source_indices": list(example.source_indices),
                "shuffled_episode_id": shuffled.episode_id,
                "shuffled_task": shuffled.task,
                "shuffled_instruction": shuffled.instruction,
                "metrics": per_sample[index],
            }
        )
    aggregate_metrics = dict(metrics)
    del aggregate_metrics["per_sample"]
    artifact = {
        "schema_version": 1,
        "step": step,
        "metrics": aggregate_metrics,
        "gate": evaluate_grounding_gates(metrics).to_dict(),
        "samples": samples,
    }
    json.dumps(artifact, sort_keys=True, allow_nan=False)
    return artifact


def publish_grounding_artifact(
    destination: str | os.PathLike[str],
    artifact: Mapping[str, Any],
) -> Path:
    """Atomically publish a fully validated grounding JSON artifact."""

    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


__all__ = (
    "GroundingExample",
    "GroundingGateDecision",
    "build_grounding_artifact",
    "compute_grounding_metrics",
    "evaluate_grounding_gates",
    "publish_grounding_artifact",
    "task_distinct_shuffle",
)
