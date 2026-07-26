from __future__ import annotations

import json

import pytest


def test_released_geometry_is_384_k4_27_square() -> None:
    from qwen35_planx.config import PlanGeometry

    geometry = PlanGeometry()

    assert geometry.image_size == 384
    assert geometry.grid_size == 27
    assert geometry.tokens_per_frame == 729
    assert geometry.tokens_per_camera == 2916
    assert geometry.tokens_per_sample == 5832
    assert geometry.visual_vocab_size == 65_536
    assert geometry.ta_code_dim == 1536
    assert geometry.qwen_hidden_dim == 2048
    assert geometry.text_align_dim == 1152
    assert geometry.ge_act_future_indices == (0, 3, 5, 8)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image_size": 256}, "image_size"),
        ({"grid_size": 16}, "grid_size"),
    ],
)
def test_released_geometry_rejects_superseded_values(
    kwargs: dict[str, object], message: str
) -> None:
    from qwen35_planx.config import PlanGeometry

    with pytest.raises(ValueError, match=message):
        PlanGeometry(**kwargs)


def test_released_ta_metadata_has_no_qwen_anchor_fields() -> None:
    from qwen35_planx.config import ReleasedTATokMetadata

    metadata = ReleasedTATokMetadata.example()
    payload = metadata.to_dict()

    assert payload["teacher"] == "google/siglip2-so400m-patch14-384"
    assert payload["checkpoint_hash"]
    assert "anchor_token_ids" not in payload
    assert "anchor_embedding_hash" not in payload
    assert ReleasedTATokMetadata.from_dict(json.loads(json.dumps(payload))) == metadata


def test_released_ta_metadata_rejects_superseded_anchor_fields() -> None:
    from qwen35_planx.config import ReleasedTATokMetadata

    payload = ReleasedTATokMetadata.example().to_dict()
    payload["anchor_token_ids"] = [0, 1]
    payload["anchor_embedding_hash"] = "superseded-anchor-sha256"

    with pytest.raises(ValueError, match="superseded.*anchor"):
        ReleasedTATokMetadata.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tokenizer_type", "ta_tok", "tokenizer_type"),
        ("teacher", "google/siglip2-large-patch16-256", "teacher"),
        ("image_size", 256, "image_size"),
        ("grid_size", 16, "grid_size"),
        ("bottleneck_token_num", 256, "bottleneck_token_num"),
        ("codebook_size", 1024, "codebook_size"),
        ("codebook_dim", 1024, "codebook_dim"),
        ("selected_layer", -1, "selected_layer"),
        ("pool_scale", 2, "pool_scale"),
    ],
)
def test_released_ta_metadata_rejects_incompatible_checkpoint_contract(
    field: str, value: object, message: str
) -> None:
    from qwen35_planx.config import ReleasedTATokMetadata

    payload = ReleasedTATokMetadata.example().to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        ReleasedTATokMetadata.from_dict(payload)


def test_hindsight_cache_rejects_missing_hashes() -> None:
    from qwen35_planx.config import HindsightCacheMetadata

    payload = HindsightCacheMetadata.example().to_dict()
    payload["dinov3_hash"] = ""

    with pytest.raises(ValueError, match="dinov3_hash"):
        HindsightCacheMetadata.from_dict(payload)


def test_grounded_planner_rejects_camera_reordering() -> None:
    from qwen35_planx.config import GroundedPlannerMetadata

    payload = GroundedPlannerMetadata.example().to_dict()
    payload["camera_names"] = ["wrist", "main"]

    with pytest.raises(ValueError, match="camera_names"):
        GroundedPlannerMetadata.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_alignment", "causal_pre_output_head_predicts_code", "hidden_alignment"),
        ("phrase_roles", ["target", "source", "action"], "phrase_roles"),
        ("hindsight_cache_hash", "", "hindsight_cache_hash"),
    ],
)
def test_grounded_planner_rejects_incompatible_grounding_contract(
    field: str, value: object, message: str
) -> None:
    from qwen35_planx.config import GroundedPlannerMetadata

    payload = GroundedPlannerMetadata.example().to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        GroundedPlannerMetadata.from_dict(payload)
