"""Fixed-global-batch Stage-1 throughput sweep."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class BatchCandidate:
    per_device_batch: int
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class TrialExecution:
    returncode: int
    metrics: Mapping[str, float] | None
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class TrialResult:
    candidate: BatchCandidate
    status: str
    metrics: Mapping[str, float] | None
    config_path: str
    error: str | None = None


@dataclass(frozen=True)
class SweepResult:
    selected: BatchCandidate | None
    trials: tuple[TrialResult, ...]


def default_worldarena_candidates(
    world_size: int,
) -> tuple[BatchCandidate, ...]:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    global_batch = 128
    candidates = tuple(
        BatchCandidate(per_device_batch, accumulation)
        for per_device_batch, accumulation in ((4, 4), (8, 2), (16, 1))
        if per_device_batch * world_size * accumulation == global_batch
    )
    if not candidates:
        raise ValueError(
            "no default WorldArena candidate preserves global batch 128"
        )
    return candidates


_REQUIRED_METRICS = (
    "throughput",
    "step_time",
    "max_memory_allocated_gib",
    "max_memory_reserved_gib",
    "device_total_memory_gib",
)


def _validated_metrics(metrics: Mapping[str, float] | None) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValueError("successful benchmark trial did not return metrics")
    validated: dict[str, float] = {}
    for name in _REQUIRED_METRICS:
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"benchmark metric {name} must be numeric")
        converted = float(value)
        if not math.isfinite(converted) or converted < 0:
            raise ValueError(f"benchmark metric {name} must be finite and non-negative")
        validated[name] = converted
    if validated["throughput"] <= 0 or validated["step_time"] <= 0:
        raise ValueError("benchmark throughput and step_time must be positive")
    if (
        validated["max_memory_allocated_gib"]
        > validated["max_memory_reserved_gib"]
        or validated["max_memory_reserved_gib"]
        > validated["device_total_memory_gib"]
    ):
        raise ValueError("benchmark memory metrics are inconsistent")
    return validated


def summarize_trial_metrics(
    path: str | Path,
    *,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, float]:
    """Aggregate an exact measured window of integrity-valid metric records."""

    from qwen35_baton.cli.train_semantic_planner import (
        _validated_durable_metrics_record,
    )

    for name, value in (
        ("warmup_steps", warmup_steps),
        ("measured_steps", measured_steps),
    ):
        if type(value) is not int or value < (0 if name == "warmup_steps" else 1):
            raise ValueError(f"{name} is invalid")
    metrics_path = Path(path)
    try:
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"benchmark metrics are unreadable: {metrics_path}") from error
    first_step = warmup_steps + 1
    final_step = warmup_steps + measured_steps
    selected: dict[int, Mapping[str, float]] = {}
    for line in lines:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        record = _validated_durable_metrics_record(raw)
        if record is None or not first_step <= record["step"] <= final_step:
            continue
        step = record["step"]
        record_metrics = record["metrics"]
        previous = selected.get(step)
        if previous is not None and previous != record_metrics:
            raise ValueError(f"conflicting valid benchmark metrics at step {step}")
        selected[step] = record_metrics
    expected_steps = list(range(first_step, final_step + 1))
    if sorted(selected) != expected_steps:
        raise ValueError(
            "benchmark metrics do not contain the complete measured window: "
            f"expected {expected_steps[0]}..{expected_steps[-1]}, "
            f"found {sorted(selected)}"
        )
    rows = [selected[step] for step in expected_steps]
    return _validated_metrics(
        {
            "throughput": median(row["throughput"] for row in rows),
            "step_time": median(row["step_time"] for row in rows),
            "max_memory_allocated_gib": max(
                row["max_memory_allocated_gib"] for row in rows
            ),
            "max_memory_reserved_gib": max(
                row["max_memory_reserved_gib"] for row in rows
            ),
            "device_total_memory_gib": min(
                row["device_total_memory_gib"] for row in rows
            ),
        }
    )


def _summary_payload(
    trials: Sequence[TrialResult],
    selected: BatchCandidate | None,
) -> dict[str, Any]:
    return {
        "selected": None if selected is None else asdict(selected),
        "trials": [
            {
                "candidate": asdict(trial.candidate),
                "status": trial.status,
                "metrics": trial.metrics,
                "config_path": trial.config_path,
                "error": trial.error,
            }
            for trial in trials
        ],
    }


def _execute_trial(
    config_path: Path,
    candidate: BatchCandidate,
    warmup_steps: int,
    measured_steps: int,
    *,
    world_size: int,
) -> TrialExecution:
    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "qwen35_baton/scripts/train_semantic_planner.sh"
    environment = os.environ.copy()
    environment.update(
        {
            "CONFIG": str(config_path),
            "NUM_GPUS": str(world_size),
            "PER_DEVICE_BATCH": str(candidate.per_device_batch),
            "GRAD_ACCUM": str(candidate.gradient_accumulation_steps),
            "GRADIENT_CHECKPOINTING": "0",
            "STOP_AT_STEP": str(warmup_steps + measured_steps),
            "PYTHON_BIN": sys.executable,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=repo_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (config_path.parent / "stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (config_path.parent / "stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    metrics = None
    if completed.returncode == 0:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = summarize_trial_metrics(
            Path(config["output_dir"]) / "training_metrics.jsonl",
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
        )
    return TrialExecution(
        returncode=completed.returncode,
        metrics=metrics,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_batch_sweep(
    base_config: Mapping[str, Any],
    candidates: Sequence[BatchCandidate],
    output_dir: str | Path,
    *,
    warmup_steps: int,
    measured_steps: int,
    command_runner: Callable[
        [Path, BatchCandidate, int, int], TrialExecution
    ]
    | None = None,
    world_size: int,
) -> SweepResult:
    """Run isolated trials and select the fastest candidate with 5 GiB headroom."""

    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    if not candidates:
        raise ValueError("benchmark sweep requires at least one candidate")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if command_runner is None:
        command_runner = lambda path, candidate, warmup, measured: _execute_trial(
            path,
            candidate,
            warmup,
            measured,
            world_size=world_size,
        )
    for name, value in (
        ("warmup_steps", warmup_steps),
        ("measured_steps", measured_steps),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    root = Path(output_dir).expanduser().resolve()
    production_output = Path(str(base_config.get("output_dir", ""))).expanduser().resolve()
    if root == production_output:
        raise ValueError("benchmark output must differ from production output_dir")
    root.mkdir(parents=True, exist_ok=False)

    trials: list[TrialResult] = []
    stable: list[tuple[BatchCandidate, float]] = []
    for candidate in candidates:
        if not isinstance(candidate, BatchCandidate):
            raise TypeError("candidates must contain BatchCandidate values")
        if (
            candidate.per_device_batch
            * world_size
            * candidate.gradient_accumulation_steps
            != 128
        ):
            raise ValueError("every benchmark candidate must preserve global batch 128")
        name = (
            f"b{candidate.per_device_batch}"
            f"_a{candidate.gradient_accumulation_steps}"
        )
        trial_root = root / name
        trial_root.mkdir()
        payload = dict(base_config)
        payload.update(
            {
                "output_dir": str(trial_root / "output"),
                "per_device_batch": candidate.per_device_batch,
                "gradient_accumulation_steps": (
                    candidate.gradient_accumulation_steps
                ),
                "resume_from": None,
                "log_every": 1,
            }
        )
        config_path = trial_root / "config.json"
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        execution = command_runner(
            config_path,
            candidate,
            warmup_steps,
            measured_steps,
        )
        if not isinstance(execution, TrialExecution):
            raise TypeError("command_runner must return TrialExecution")
        if execution.returncode != 0:
            error_text = "\n".join((execution.stdout, execution.stderr)).strip()
            status = (
                "oom" if "out of memory" in error_text.lower() else "failed"
            )
            trials.append(
                TrialResult(
                    candidate=candidate,
                    status=status,
                    metrics=None,
                    config_path=str(config_path),
                    error=error_text,
                )
            )
            if status == "failed":
                (root / "summary.json").write_text(
                    json.dumps(_summary_payload(trials, None), indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError(f"benchmark trial {name} failed: {error_text}")
            continue
        metrics = _validated_metrics(execution.metrics)
        headroom = (
            metrics["device_total_memory_gib"]
            - metrics["max_memory_reserved_gib"]
        )
        status = "stable" if headroom >= 5.0 else "insufficient_headroom"
        trials.append(
            TrialResult(
                candidate=candidate,
                status=status,
                metrics=metrics,
                config_path=str(config_path),
            )
        )
        if status == "stable":
            stable.append((candidate, metrics["throughput"]))

    selected = max(stable, key=lambda item: item[1])[0] if stable else None
    result = SweepResult(selected=selected, trials=tuple(trials))
    (root / "summary.json").write_text(
        json.dumps(_summary_payload(result.trials, selected), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"benchmark config is unreadable: {args.config}") from error
    result = run_batch_sweep(
        payload,
        default_worldarena_candidates(args.world_size),
        args.output_dir,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        world_size=args.world_size,
    )
    print(
        json.dumps(
            _summary_payload(result.trials, result.selected),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
