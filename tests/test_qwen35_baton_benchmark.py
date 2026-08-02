from __future__ import annotations

import json


def test_worldarena_sweep_preserves_global_batch_128() -> None:
    from qwen35_baton.cli.benchmark_stage1_throughput import (
        default_worldarena_candidates,
    )

    candidates = default_worldarena_candidates(world_size=8)

    assert [
        (item.per_device_batch, item.gradient_accumulation_steps)
        for item in candidates
    ] == [(4, 4), (8, 2), (16, 1)]
    assert all(
        item.per_device_batch
        * 8
        * item.gradient_accumulation_steps
        == 128
        for item in candidates
    )


def test_sweep_records_oom_and_selects_fastest_stable_candidate(tmp_path) -> None:
    from qwen35_baton.cli.benchmark_stage1_throughput import (
        TrialExecution,
        default_worldarena_candidates,
        run_batch_sweep,
    )

    outcomes = {
        (4, 4): TrialExecution(
            returncode=0,
            metrics={
                "throughput": 70.0,
                "step_time": 1.8,
                "max_memory_allocated_gib": 50.0,
                "max_memory_reserved_gib": 52.0,
                "device_total_memory_gib": 80.0,
            },
        ),
        (8, 2): TrialExecution(
            returncode=0,
            metrics={
                "throughput": 91.0,
                "step_time": 1.4,
                "max_memory_allocated_gib": 65.0,
                "max_memory_reserved_gib": 68.0,
                "device_total_memory_gib": 80.0,
            },
        ),
        (16, 1): TrialExecution(
            returncode=1,
            metrics=None,
            stderr="torch.OutOfMemoryError: CUDA out of memory",
        ),
    }

    def runner(config_path, candidate, warmup_steps, measured_steps):
        assert config_path.is_file()
        assert warmup_steps == 5
        assert measured_steps == 20
        return outcomes[
            (candidate.per_device_batch, candidate.gradient_accumulation_steps)
        ]

    result = run_batch_sweep(
        {
            "output_dir": str(tmp_path / "production"),
            "per_device_batch": 4,
            "gradient_accumulation_steps": 4,
            "resume_from": None,
        },
        default_worldarena_candidates(8),
        tmp_path / "benchmark",
        warmup_steps=5,
        measured_steps=20,
        command_runner=runner,
        world_size=8,
    )

    assert result.selected is not None
    assert result.selected.per_device_batch == 8
    assert result.trials[-1].status == "oom"
    summary = json.loads((tmp_path / "benchmark/summary.json").read_text())
    assert summary["selected"]["per_device_batch"] == 8
    assert summary["trials"][-1]["status"] == "oom"


def test_trial_summary_uses_only_integrity_valid_measured_steps(tmp_path) -> None:
    from qwen35_baton.cli.benchmark_stage1_throughput import (
        summarize_trial_metrics,
    )
    from qwen35_baton.cli.train_semantic_planner import _durable_metrics_record

    def metrics(throughput: float, step_time: float) -> dict[str, float]:
        return {
            "loss/total": 1.0,
            "loss/mse": 1.0,
            "data_time": 0.1,
            "qwen_time": 0.2,
            "teacher_time": 0.3,
            "query_tower_time": 0.4,
            "backward_time": 0.5,
            "throughput": throughput,
            "microbatches": 1.0,
            "step_time": step_time,
            "max_memory_allocated_gib": 60.0,
            "max_memory_reserved_gib": 64.0,
            "device_total_memory_gib": 80.0,
            **{f"mse/head/frame_{frame}": 1.0 for frame in range(4)},
        }

    path = tmp_path / "training_metrics.jsonl"
    records = [
        _durable_metrics_record(step=1, metrics=metrics(10.0, 10.0)),
        _durable_metrics_record(step=2, metrics=metrics(80.0, 1.6)),
        _durable_metrics_record(step=3, metrics=metrics(100.0, 1.2)),
    ]
    corrupt = dict(records[-1])
    corrupt["checksum"] = "0" * 64
    path.write_text(
        "\n".join(json.dumps(record) for record in [*records, corrupt]) + "\n"
    )

    summary = summarize_trial_metrics(path, warmup_steps=1, measured_steps=2)

    assert summary["throughput"] == 90.0
    assert summary["step_time"] == 1.4
    assert summary["max_memory_reserved_gib"] == 64.0
