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
    BatonTemporalPolicy,
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
    assert worldarena["runtime_input_validation"] is False
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
        "temporal_policy",
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
    assert payload["format_version"] == 4
    assert payload["temporal_policy"] == {
        "kind": "fixed_offsets",
        "offsets": [0, 3, 5, 8],
    }
    assert "future_indices" not in payload
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
    payload = metadata.to_dict()

    assert metadata.camera_names == ("head",)
    assert metadata.target_shape == (1, 4, 256, 1024)
    assert metadata.temporal_policy == BatonTemporalPolicy.worldarena_normalized()
    assert payload["temporal_policy"] == {
        "kind": "normalized_remaining_horizon",
        "canonical_frame_count": 121,
        "current_index_range": [0, 116],
        "target_count": 4,
        "rounding": "round_half_even_exact_integer_v1",
        "formula": ("f_k = c + round_half_even((k + 1) * (120 - c) / 4), k=0..3"),
    }
    assert "future_indices" not in payload
    assert [0, 3, 5, 8] not in payload.values()
    assert BatonCheckpointMetadata.from_dict(payload) == metadata


def _legacy_v3_payload(metadata: BatonCheckpointMetadata) -> dict[str, object]:
    payload = metadata.to_dict()
    payload["format_version"] = 3
    payload["future_indices"] = [0, 3, 5, 8]
    del payload["temporal_policy"]
    return payload


def test_legacy_v3_libero_metadata_migrates_in_memory_without_rewrite() -> None:
    restored = BatonCheckpointMetadata.from_dict(
        _legacy_v3_payload(BatonCheckpointMetadata.example())
    )

    assert restored.format_version == 4
    assert restored.camera_names == ("main", "wrist")
    assert restored.temporal_policy == BatonTemporalPolicy.libero_fixed()


def test_legacy_v3_head_metadata_requires_explicit_checkpoint_migration() -> None:
    payload = _legacy_v3_payload(
        BatonCheckpointMetadata.example(camera_names=("head",))
    )

    with pytest.raises(ValueError, match="legacy head.*migration required"):
        BatonCheckpointMetadata.from_dict(payload)


def test_temporal_policy_uses_exact_half_even_remaining_horizon_contract() -> None:
    policy = BatonTemporalPolicy.worldarena_normalized()

    assert policy.resolve_future_indices(current_index=0) == (30, 60, 90, 120)
    assert policy.resolve_future_indices(current_index=2) == (32, 61, 90, 120)
    assert policy.resolve_future_indices(current_index=116) == (117, 118, 119, 120)
    with pytest.raises(ValueError, match="current_index"):
        policy.resolve_future_indices()
    with pytest.raises(ValueError, match=r"\[0, 116\]"):
        policy.resolve_future_indices(current_index=117)


def test_metadata_rejects_camera_and_temporal_policy_reinterpretation() -> None:
    libero = BatonCheckpointMetadata.example().to_dict()
    libero["temporal_policy"] = BatonTemporalPolicy.worldarena_normalized().to_dict()
    with pytest.raises(ValueError, match="camera_names.*temporal_policy"):
        BatonCheckpointMetadata.from_dict(libero)

    head = BatonCheckpointMetadata.example(camera_names=("head",)).to_dict()
    head["temporal_policy"] = BatonTemporalPolicy.libero_fixed().to_dict()
    with pytest.raises(ValueError, match="camera_names.*temporal_policy"):
        BatonCheckpointMetadata.from_dict(head)


@pytest.mark.parametrize(
    ("camera_names", "mutate", "message"),
    (
        (
            ("main", "wrist"),
            lambda policy: policy["offsets"].__setitem__(0, False),
            "offsets",
        ),
        (
            ("head",),
            lambda policy: policy.__setitem__("canonical_frame_count", 121.0),
            "canonical_frame_count",
        ),
        (
            ("head",),
            lambda policy: policy["current_index_range"].__setitem__(0, False),
            "current_index_range",
        ),
        (
            ("head",),
            lambda policy: policy.__setitem__("target_count", 4.0),
            "target_count",
        ),
    ),
)
def test_v4_temporal_policy_rejects_noncanonical_json_integer_types(
    camera_names: tuple[str, ...],
    mutate: object,
    message: str,
) -> None:
    payload = BatonCheckpointMetadata.example(camera_names=camera_names).to_dict()
    assert callable(mutate)
    mutate(payload["temporal_policy"])

    with pytest.raises((TypeError, ValueError), match=message):
        BatonCheckpointMetadata.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("format_version", 4.0),
        ("teacher_image_size", 256.0),
        ("teacher_feature_layer", -2.0),
        ("query_dim", 2048.0),
        ("global_step", False),
    ),
)
def test_v4_metadata_rejects_noncanonical_json_integer_types(
    field: str,
    value: object,
) -> None:
    payload = BatonCheckpointMetadata.example().to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        BatonCheckpointMetadata.from_dict(payload)


def test_v4_metadata_rejects_noncanonical_integer_sequence_elements() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()
    payload["target_shape"][1] = 4.0

    with pytest.raises((TypeError, ValueError), match="target_shape"):
        BatonCheckpointMetadata.from_dict(payload)


def test_continuous_metadata_rejects_wrong_backbone_identity() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()

    payload["qwen_backbone"] = "Qwen3.5-2B-MoE"

    with pytest.raises(ValueError, match="qwen_backbone"):
        BatonCheckpointMetadata.from_dict(payload)


def test_format_v1_metadata_is_explicitly_incompatible_with_v4() -> None:
    payload = BatonCheckpointMetadata.example().to_dict()
    payload["format_version"] = 1

    with pytest.raises(ValueError, match="versions 1 and 2.*version 4"):
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
    assert (
        sha256_file(artifact, chunk_size=3)
        == hashlib.sha256(b"baton-contract\n").hexdigest()
    )


def test_baton_package_does_not_import_legacy_planners() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO_ROOT / "qwen35_baton").rglob("*.py"))
    )

    assert "qwen35_planx" not in source
    assert "qwen3_vl_semantic_planner" not in source
