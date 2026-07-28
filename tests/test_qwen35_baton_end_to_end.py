from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from qwen35_baton.cli.smoke_pipeline import (
    _invocation_directory,
    run_tiny_pipeline,
    validate_two_rank_result,
)


@pytest.fixture(scope="module")
def tiny_pipeline_result(tmp_path_factory: pytest.TempPathFactory):
    return run_tiny_pipeline(tmp_path_factory.mktemp("baton-e2e"))


def test_tiny_pipeline_runs_all_three_production_boundaries(
    tiny_pipeline_result,
) -> None:
    result = tiny_pipeline_result

    assert result.stage1.optimizer_steps == 1
    assert result.stage2.optimizer_steps == 1
    assert result.stage3.optimizer_steps == 1
    assert result.stage1.plan_shape == (1, 2, 4, 256, 1024)
    assert result.stage2.condition_source == "teacher"
    assert result.stage3.condition_source == "prediction"
    assert result.stage1.source_ownership == "frozen_siglip2_teacher"
    assert result.stage2.source_ownership == "frozen_siglip2_teacher"
    assert result.stage3.source_ownership == "frozen_baton_prediction"
    assert result.stage1.source_hash_before == result.stage1.source_hash_after
    assert result.stage2.source_hash_before == result.stage2.source_hash_after
    assert result.stage3.source_hash_before == result.stage3.source_hash_after
    assert result.stage1.trainable_hash_before != result.stage1.trainable_hash_after
    assert result.stage2.trainable_hash_before != result.stage2.trainable_hash_after
    assert result.stage3.trainable_hash_before != result.stage3.trainable_hash_after


def test_stage3_strictly_loads_the_stage2_baton_checkpoint_artifact(
    tiny_pipeline_result,
) -> None:
    result = tiny_pipeline_result

    assert result.checkpoint.source == "qwen35_baton_teacher"
    assert result.checkpoint.cursor == {
        "global_step": 1,
        "epoch": 0,
        "consumed_microbatches": 1,
        "microbatches_per_epoch": 4,
        "sampler_seed": 42,
    }
    assert result.checkpoint.envelope_loaded
    assert result.checkpoint.strict_stage3_loaded
    assert result.checkpoint.stage2_artifact_hash == result.stage2.trainable_hash_after
    assert result.stage3.trainable_hash_before == result.stage2.trainable_hash_after
    assert result.checkpoint.optimizer_hash
    assert result.checkpoint.scheduler_hash
    assert result.checkpoint.rng_hash


def test_single_rank_smoke_reports_rank_execution_without_fake_resume(
    tiny_pipeline_result,
) -> None:
    result = tiny_pipeline_result

    assert result.rank_agreement
    assert result.executed_ranks == (0,)
    assert result.exact_resume is None
    assert result.fresh_process_restore is None


def test_two_rank_result_validation_rejects_a_false_green(
    tmp_path,
) -> None:
    incomplete = tmp_path / "result.json"
    incomplete.write_text(
        '{"rank_agreement": true, "exact_resume": false}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        validate_two_rank_result(incomplete)


@pytest.mark.parametrize(
    "invalid",
    [
        "invocation-not-hex",
        "invocation-" + "a" * 31,
        "invocation-" + "a" * 33,
        "invocation-" + "A" * 32,
        "prefix-invocation-" + "a" * 32,
    ],
)
def test_invocation_directory_requires_exact_canonical_identifier(
    tmp_path,
    monkeypatch,
    invalid: str,
) -> None:
    monkeypatch.setenv("QWEN35_BATON_SMOKE_INVOCATION_ID", invalid)

    with pytest.raises(ValueError, match="invocation ID"):
        _invocation_directory(tmp_path)


def test_two_rank_smoke_detects_a_missing_gradient_collective(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ)
    environment["QWEN35_BATON_SMOKE_INVOCATION_ID"] = (
        "invocation-" + "a" * 32
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            "qwen35_baton.cli.smoke_pipeline",
            "--output-dir",
            str(tmp_path),
            "--disable-gradient-sync",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
        env=environment,
    )

    assert completed.returncode != 0
    assert "synchronized parameter hashes differ by rank" in (
        completed.stdout + completed.stderr
    )
