"""Contract tests for the dependency-light continuous Baton package."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from qwen35_baton import (
    BatonCheckpointMetadata,
    BatonGeometry,
    BatonLossWeights,
    sha256_file,
    sha256_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_baton_geometry_is_the_approved_256px_contract() -> None:
    geometry = BatonGeometry()

    assert geometry.camera_names == ("main", "wrist")
    assert geometry.future_indices == (0, 3, 5, 8)
    assert geometry.image_size == 256
    assert geometry.grid_size == 16
    assert geometry.tokens_per_frame == 256
    assert geometry.tokens_per_camera == 1024
    assert geometry.output_shape(3) == (3, 2, 4, 256, 1024)


def test_continuous_metadata_round_trips_the_complete_contract() -> None:
    metadata = BatonCheckpointMetadata.example()

    payload = metadata.to_dict()

    assert set(payload) == {
        "format_version",
        "architecture_kind",
        "qwen_config_hash",
        "tokenizer_hash",
        "processor_hash",
        "input_template_hash",
        "added_tokens",
        "added_token_ids",
        "camera_names",
        "camera_flattening",
        "siglip2_model",
        "siglip2_artifact_hash",
        "teacher_image_size",
        "teacher_patch_size",
        "teacher_feature_layer",
        "teacher_preprocessing_hash",
        "teacher_dtype",
        "target_shape",
        "future_indices",
        "query_dim",
        "query_layers",
        "query_heads",
        "query_ffn_dim",
        "query_dropout",
        "query_mask_version",
        "trainable_qwen_layer_indices",
        "loss_weights",
        "hdf5_manifest_hash",
        "optimizer_topology_hash",
        "scheduler_topology_hash",
        "global_step",
        "distributed_cursor",
        "rng_state_hash",
    }
    assert payload["architecture_kind"] == "qwen35_baton_continuous"
    assert payload["target_shape"] == [2, 4, 256, 1024]
    assert payload["future_indices"] == [0, 3, 5, 8]
    assert payload["teacher_feature_layer"] == -2
    assert payload["loss_weights"] == {
        "mse": 1.0,
        "cosine": 0.5,
        "delta": 0.5,
        "instruction_counterfactual": 0.2,
        "counterfactual_margin": 0.1,
    }
    assert BatonCheckpointMetadata.from_dict(payload) == metadata


def test_continuous_metadata_rejects_discrete_checkpoints() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()
    payload["architecture_kind"] = "qwen35_planx_grounded"

    with pytest.raises(ValueError, match="qwen35_baton_continuous"):
        BatonCheckpointMetadata.from_dict(payload)


def test_continuous_metadata_rejects_invalid_fixed_geometry() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()
    payload["target_shape"] = [2, 4, 729, 1536]

    with pytest.raises(ValueError, match="target_shape"):
        BatonCheckpointMetadata.from_dict(payload)


def test_loss_weights_are_the_approved_stage1_values() -> None:
    assert BatonLossWeights() == BatonLossWeights(
        mse=1.0,
        cosine=0.5,
        delta=0.5,
        instruction_counterfactual=0.2,
        counterfactual_margin=0.1,
    )


def test_sha256_helpers_are_stable_and_stream_file_contents(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"baton-contract\n")

    assert sha256_json({"b": [2, 1], "a": "Baton"}) == sha256_json(
        {"a": "Baton", "b": [2, 1]}
    )
    assert sha256_file(artifact, chunk_size=3) == hashlib.sha256(
        b"baton-contract\n"
    ).hexdigest()


def test_baton_package_does_not_import_legacy_planners() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "qwen35_baton").rglob("*.py"))
    )

    assert "qwen35_planx" not in source
    assert "qwen3_vl_semantic_planner" not in source
