"""Contract tests for the dependency-light continuous Baton package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from qwen35_baton import (
    BatonCheckpointMetadata,
    BatonGeometry,
    sha256_file,
    sha256_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_worldarena_stage1_recipe_preserves_model_provenance_and_global_batch() -> None:
    libero = json.loads(
        (REPO_ROOT / "qwen35_baton/configs/libero_stage1.json").read_text()
    )
    worldarena = json.loads(
        (REPO_ROOT / "qwen35_baton/configs/worldarena_stage1.json").read_text()
    )

    assert worldarena["dataset_type"] == "worldarena_hdf5"
    assert (
        worldarena["per_device_batch"] * 8 * worldarena["gradient_accumulation_steps"]
        == 128
    )
    assert worldarena["per_device_batch"] == 2
    assert worldarena["gradient_accumulation_steps"] == 8
    assert worldarena["max_steps"] == 30_000
    assert worldarena["initial_save_step"] == 20
    assert worldarena["save_every"] == 5_000
    assert worldarena["num_workers"] == 8
    assert worldarena["persistent_workers"] is True
    assert worldarena["worker_restart_interval_epochs"] == 100
    for field in (
        "qwen_model_path",
        "qwen_processor_path",
        "qwen_tokenizer_path",
        "siglip2_model_path",
        "siglip2_config_hash",
        "siglip2_artifact_hash",
        "deepspeed_config_path",
    ):
        assert worldarena[field] == libero[field]


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
        "qwen_backbone",
        "qwen_config_hash",
        "tokenizer_hash",
        "processor_hash",
        "input_template_hash",
        "added_tokens",
        "added_token_ids",
        "camera_names",
        "camera_flattening",
        "siglip2_model",
        "siglip2_config_hash",
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
        "query_norm_style",
        "query_mask_version",
        "trainable_qwen_layer_indices",
        "loss_weights",
        "hdf5_manifest_hash",
        "planner_topology_hash",
        "optimizer_topology_hash",
        "scheduler_topology_hash",
        "global_step",
        "distributed_cursor",
        "rng_state_hash",
    }
    assert payload["architecture_kind"] == "qwen35_baton_continuous"
    assert payload["qwen_backbone"] == "dense Qwen3.5-2B"
    assert payload["trainable_qwen_layer_indices"] == list(range(24))
    assert payload["target_shape"] == [2, 4, 256, 1024]
    assert payload["future_indices"] == [0, 3, 5, 8]
    assert payload["teacher_feature_layer"] == -2
    assert payload["query_dim"] == 2048
    assert payload["query_layers"] == 1
    assert payload["query_ffn_dim"] == 0
    assert payload["query_dropout"] == 0.0
    assert payload["query_norm_style"] == "none"
    assert payload["query_mask_version"] == "full_cross_attention_v1"
    assert payload["loss_weights"] == {
        "mse": 1.0,
    }
    assert BatonCheckpointMetadata.from_dict(payload) == metadata


def test_checkpoint_metadata_accepts_truthful_head_camera_shape() -> None:
    metadata = BatonCheckpointMetadata.example(camera_names=("head",))
    assert metadata.camera_names == ("head",)
    assert metadata.target_shape == (1, 4, 256, 1024)
    assert BatonCheckpointMetadata.from_dict(metadata.to_dict()) == metadata


def test_continuous_metadata_rejects_wrong_backbone_identity() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()

    payload["qwen_backbone"] = "Qwen3.5-2B-MoE"

    with pytest.raises(ValueError, match="qwen_backbone"):
        BatonCheckpointMetadata.from_dict(payload)


def test_format_v1_metadata_is_explicitly_incompatible_with_v2() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()
    payload["format_version"] = 1

    with pytest.raises(ValueError, match="versions 1 and 2.*version 3"):
        BatonCheckpointMetadata.from_dict(payload)


def test_continuous_metadata_rejects_wrong_teacher_identity() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()

    payload["siglip2_model"] = "SigLIP2-large-patch14-384"

    with pytest.raises(ValueError, match="siglip2_model"):
        BatonCheckpointMetadata.from_dict(payload)


@pytest.mark.parametrize("field", BatonCheckpointMetadata._HASH_FIELDS)
def test_continuous_metadata_rejects_non_sha256_provenance(field: str) -> None:
    payload = BatonCheckpointMetadata.example().to_dict()

    payload[field] = "not-a-sha256"

    with pytest.raises(ValueError, match=field):
        BatonCheckpointMetadata.from_dict(payload)


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


def test_continuous_metadata_requires_pre_normalized_query_blocks() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()

    payload["query_norm_style"] = "post_norm"

    with pytest.raises(ValueError, match="query_norm_style"):
        BatonCheckpointMetadata.from_dict(payload)


def test_continuous_metadata_rejects_non_integer_added_token_ids() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()
    payload["added_token_ids"][0] = True

    with pytest.raises(ValueError, match="added_token_ids"):
        BatonCheckpointMetadata.from_dict(payload)


def test_continuous_metadata_rejects_duplicate_distributed_cursor_names() -> None:
    metadata = BatonCheckpointMetadata.example()

    with pytest.raises(ValueError, match="distributed_cursor"):
        replace(metadata, distributed_cursor=(("epoch", 0), ("epoch", 1)))


def test_distributed_cursor_round_trips_without_loss() -> None:
    metadata = replace(
        BatonCheckpointMetadata.example(),
        distributed_cursor=(
            ("epoch", 2),
            ("consumed_microbatches", 17),
            ("sampler_seed", 9),
        ),
    )

    restored = BatonCheckpointMetadata.from_dict(metadata.to_dict())

    assert restored.distributed_cursor == metadata.distributed_cursor


def test_metadata_carries_only_baton_equation_8_mse() -> None:
    assert BatonCheckpointMetadata.example().loss_weights == {"mse": 1.0}


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
