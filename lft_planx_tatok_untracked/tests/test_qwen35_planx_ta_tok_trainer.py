from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_split_isolation_rejects_shared_trajectory() -> None:
    from qwen35_planx.ta_tok_trainer import validate_split_isolation

    train = [{"trajectory_id": "suite:000001"}]
    val = [{"trajectory_id": "suite:000001"}]
    with pytest.raises(ValueError, match="trajectory"):
        validate_split_isolation(train, val)


def test_validation_summary_requires_and_reports_both_cameras() -> None:
    from qwen35_planx.ta_tok_trainer import summarize_validation

    batches = [
        {"camera": "main", "reconstruction_cosine": 0.8},
        {"camera": "wrist", "reconstruction_cosine": 0.6},
        {"camera": "main", "reconstruction_cosine": 1.0},
    ]
    summary = summarize_validation(batches)
    assert summary["main"]["reconstruction_cosine"] == pytest.approx(0.9)
    assert summary["wrist"]["reconstruction_cosine"] == pytest.approx(0.6)
    assert summary["overall"]["reconstruction_cosine"] == pytest.approx(0.8)

    with pytest.raises(ValueError, match="wrist"):
        summarize_validation(batches[:1])


def _metadata():
    from qwen35_planx.config import TATokMetadata

    return TATokMetadata.example()


def _metrics(*, coverage: float = 0.25, dead: float = 0.75):
    camera = {
        "reconstruction_cosine": 0.7,
        "coverage": coverage,
        "perplexity": 100.0,
        "dead_code_ratio": dead,
        "commitment": 0.1,
        "codebook": 0.2,
    }
    return {"main": camera, "wrist": camera, "overall": camera}


def test_checkpoint_is_hash_bound_atomic_and_omits_teacher(tmp_path) -> None:
    from qwen35_planx.ta_tok_trainer import save_ta_tok_checkpoint

    state = {
        "student.weight": torch.ones(2, 2),
        "teacher.weight": torch.full((2, 2), 9.0),
    }
    first = save_ta_tok_checkpoint(
        output_dir=tmp_path,
        step=10,
        state_dict=state,
        metadata=_metadata(),
        anchor_token_ids=np.arange(65_536, dtype=np.int64),
        metrics=_metrics(),
    )
    second = save_ta_tok_checkpoint(
        output_dir=tmp_path,
        step=11,
        state_dict={**state, "student.weight": torch.zeros(2, 2)},
        metadata=_metadata(),
        anchor_token_ids=np.arange(65_536, dtype=np.int64),
        metrics=_metrics(),
    )

    assert first.state_hash != second.state_hash
    assert (first.path / "ta_tok.safetensors").is_file()
    assert (first.path / "metadata.json").is_file()
    assert (first.path / "anchor_ids.npy").is_file()
    assert (first.path / "metrics.json").is_file()
    assert (tmp_path / "latest_checkpoint.txt").read_text().strip() == first.path.name.replace("000010", "000011")

    from safetensors.torch import load_file

    saved = load_file(first.path / "ta_tok.safetensors")
    assert "student.weight" in saved
    assert not any(key.startswith("teacher.") for key in saved)


def test_resume_and_preflight_reject_anchor_or_collapse_mismatch(tmp_path) -> None:
    from qwen35_planx.cli.preflight import preflight_ta_tok_checkpoint
    from qwen35_planx.ta_tok_trainer import (
        save_ta_tok_checkpoint,
        validate_resume_anchors,
    )

    expected = np.arange(65_536, dtype=np.int64)
    with pytest.raises(ValueError, match="anchor"):
        validate_resume_anchors(expected, expected[::-1].copy())

    checkpoint = save_ta_tok_checkpoint(
        output_dir=tmp_path,
        step=1,
        state_dict={"student.weight": torch.ones(1)},
        metadata=_metadata(),
        anchor_token_ids=expected,
        metrics=_metrics(coverage=0.01, dead=0.99),
    )
    with pytest.raises(ValueError, match="coverage"):
        preflight_ta_tok_checkpoint(
            checkpoint.path,
            min_coverage=0.1,
            max_dead_code_ratio=0.95,
        )

    metadata_path = checkpoint.path / "metadata.json"
    payload = json.loads(metadata_path.read_text())
    payload["selected_layer"] = -1
    metadata_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="selected_layer"):
        preflight_ta_tok_checkpoint(
            checkpoint.path,
            min_coverage=0.0,
            max_dead_code_ratio=1.0,
        )


def test_frame_manifest_dataset_loads_exact_predecoded_camera_frame(
    tmp_path: Path,
) -> None:
    from qwen35_planx.ta_tok_trainer import (
        FrameManifestDataset,
        load_frame_manifest,
    )

    cache = tmp_path / "camera.npy"
    frames = np.zeros((3, 6, 8, 3), dtype=np.uint8)
    frames[2, :, :, 0] = 255
    np.save(cache, frames, allow_pickle=False)
    manifest = tmp_path / "ta_frames_train.jsonl"
    rows = [
        {
            "trajectory_id": "suite:000001",
            "suite": "suite",
            "split": "train",
            "instruction": "pick",
            "camera": "wrist",
            "frame_index": 2,
            "cache_path": str(cache),
        }
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    restored = load_frame_manifest(manifest, expected_split="train")
    sample = FrameManifestDataset(restored)[0]

    assert sample["image"].shape == (3, 6, 8)
    assert sample["image"].dtype == torch.float32
    assert sample["image"][0].mean().item() == 1.0
    assert sample["camera_id"].item() == 1
    assert sample["frame_index"].item() == 2


def test_frame_manifest_rejects_wrong_split_or_camera(tmp_path: Path) -> None:
    from qwen35_planx.ta_tok_trainer import load_frame_manifest

    manifest = tmp_path / "frames.jsonl"
    row = {
        "trajectory_id": "suite:000001",
        "suite": "suite",
        "split": "val",
        "instruction": "pick",
        "camera": "side",
        "frame_index": 0,
        "cache_path": "/missing.npy",
    }
    manifest.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="split"):
        load_frame_manifest(manifest, expected_split="train")
    with pytest.raises(ValueError, match="camera"):
        load_frame_manifest(manifest, expected_split="val")


def test_training_config_derives_global_batch_and_rejects_mismatch() -> None:
    from qwen35_planx.cli.train_ta_tok import TrainConfig

    config = TrainConfig(
        per_device_batch_size=8,
        gradient_accumulation_steps=4,
        world_size=8,
        expected_global_batch_size=256,
    )
    assert config.global_batch_size == 256

    with pytest.raises(ValueError, match="global batch"):
        TrainConfig(
            per_device_batch_size=8,
            gradient_accumulation_steps=2,
            world_size=8,
            expected_global_batch_size=256,
        )
