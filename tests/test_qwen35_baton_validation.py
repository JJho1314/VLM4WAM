from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def _examples():
    from qwen35_baton.validation import GroundingExample

    return tuple(
        GroundingExample(
            episode_id=f"task_{index % 4}__episode{index}",
            task=f"task_{index % 4}",
            instruction=f"instruction {index}",
            source_indices=(0, 30, 60, 90, 120),
        )
        for index in range(44)
    )


def test_task_distinct_shuffle_is_deterministic_bijective_and_cross_task() -> None:
    from qwen35_baton.validation import task_distinct_shuffle

    examples = _examples()
    first = task_distinct_shuffle(examples, seed=42)
    second = task_distinct_shuffle(examples, seed=42)

    assert first == second
    assert sorted(first) == list(range(44))
    assert all(examples[index].task != examples[other].task for index, other in enumerate(first))


def test_task_distinct_shuffle_rejects_an_impossible_single_task() -> None:
    from qwen35_baton.validation import GroundingExample, task_distinct_shuffle

    examples = tuple(
        GroundingExample(
            episode_id=f"same__episode{index}",
            task="same",
            instruction=f"instruction {index}",
            source_indices=(0, 30, 60, 90, 120),
        )
        for index in range(2)
    )

    with pytest.raises(ValueError, match="task-distinct shuffle"):
        task_distinct_shuffle(examples, seed=0)


def test_grounding_metrics_cover_aggregate_horizons_wins_and_scale() -> None:
    from qwen35_baton.validation import compute_grounding_metrics

    base = torch.arange(1, 5, dtype=torch.float32).view(1, 4, 1, 1)
    spatial = torch.tensor([0.5, 1.5], dtype=torch.float32).view(1, 1, 2, 1)
    channels = torch.tensor([0.75, 1.0, 1.25], dtype=torch.float32).view(1, 1, 1, 3)
    target = (base * spatial * channels).expand(44, -1, -1, -1).clone()
    correct = target * 0.90
    shuffled = target * 0.20
    persistence = torch.zeros_like(target)

    metrics = compute_grounding_metrics(
        correct=correct,
        shuffled=shuffled,
        target=target,
        persistence=persistence,
    )

    assert metrics["example_count"] == 44
    assert metrics["finite_example_count"] == 44
    assert metrics["correct_win_count"] == 44
    assert metrics["correct_win_fraction"] == 1.0
    assert metrics["shuffle_mse_improvement"] > 0.98
    assert metrics["persistence_mse_improvement"] > 0.98
    assert metrics["prediction_target_norm_ratio"] == pytest.approx(0.90)
    assert metrics["prediction_target_spatial_std_ratio"] == pytest.approx(0.90)
    assert metrics["future_delta_cosine"] == pytest.approx(1.0)
    assert len(metrics["per_horizon"]) == 4
    assert len(metrics["per_sample"]) == 44
    assert all(row["correct_mse"] < row["shuffled_mse"] for row in metrics["per_sample"])


def test_grounding_metrics_marks_nonfinite_examples_without_hiding_them() -> None:
    from qwen35_baton.validation import compute_grounding_metrics

    target = torch.ones((44, 4, 2, 3), dtype=torch.float32)
    correct = target.clone()
    correct[7, 0, 0, 0] = float("nan")
    metrics = compute_grounding_metrics(
        correct=correct,
        shuffled=torch.zeros_like(target),
        target=target,
        persistence=torch.zeros_like(target),
    )

    assert metrics["example_count"] == 44
    assert metrics["finite_example_count"] == 43
    assert metrics["per_sample"][7]["finite"] is False


def test_grounding_gate_decision_uses_all_approved_thresholds() -> None:
    from qwen35_baton.validation import evaluate_grounding_gates

    metrics = {
        "example_count": 44,
        "finite_example_count": 44,
        "correct_win_fraction": 0.60,
        "shuffle_mse_improvement": 0.05,
        "persistence_mse_improvement": 0.25,
        "prediction_target_norm_ratio": 0.85,
    }
    decision = evaluate_grounding_gates(metrics)

    assert decision.eligible is True
    assert decision.failures == ()
    assert decision.required_examples == 44
    assert decision.minimum_correct_win_fraction == 0.60
    assert decision.minimum_shuffle_mse_improvement == 0.05
    assert decision.minimum_persistence_mse_improvement == 0.25
    assert decision.accepted_norm_ratio == (0.85, 1.15)

    failed = evaluate_grounding_gates({**metrics, "correct_win_fraction": 0.59})
    assert failed.eligible is False
    assert failed.failures == ("correct_win_fraction",)


def test_grounding_artifact_is_auditable_and_atomically_published(
    tmp_path: Path,
) -> None:
    from qwen35_baton.validation import (
        build_grounding_artifact,
        publish_grounding_artifact,
        task_distinct_shuffle,
    )

    examples = _examples()
    shuffle = task_distinct_shuffle(examples, seed=42)
    metrics = {
        "example_count": 44,
        "finite_example_count": 44,
        "correct_win_fraction": 0.75,
        "shuffle_mse_improvement": 0.10,
        "persistence_mse_improvement": 0.40,
        "prediction_target_norm_ratio": 0.95,
        "per_sample": [{"finite": True, "correct_mse": 0.1}] * 44,
    }
    artifact = build_grounding_artifact(
        step=500,
        examples=examples,
        shuffled_indices=shuffle,
        metrics=metrics,
    )
    destination = tmp_path / "validation" / "step_000500.json"

    publish_grounding_artifact(destination, artifact)
    restored = json.loads(destination.read_text(encoding="utf-8"))

    assert restored["step"] == 500
    assert restored["gate"]["eligible"] is True
    assert len(restored["samples"]) == 44
    assert restored["samples"][0]["task"] != restored["samples"][0]["shuffled_task"]
    assert not tuple(destination.parent.glob(".*.tmp"))
