from __future__ import annotations

import json

import pytest


def test_default_geometry_derives_the_dual_camera_k4_contract() -> None:
    from qwen35_planx.config import CAMERA_KEYS, CAMERA_NAMES, PlanGeometry

    geometry = PlanGeometry()

    assert CAMERA_NAMES == ("main", "wrist")
    assert CAMERA_KEYS == (
        "observation.images.image",
        "observation.images.wrist_image",
    )
    assert geometry.future_frame_offsets == (1, 4, 6, 9)
    assert geometry.ge_act_future_indices == (0, 3, 5, 8)
    assert geometry.tokens_per_frame == 729
    assert geometry.tokens_per_camera == 2916
    assert geometry.tokens_per_sample == 5832
    assert geometry.response_tokens_per_camera == 2926


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_keyframes": 3}, "num_keyframes"),
        ({"image_size": 256}, "image_size"),
        ({"grid_size": 16}, "grid_size"),
        ({"visual_vocab_size": 2048}, "visual_vocab_size"),
        ({"future_frame_offsets": (1, 3, 5, 7)}, "future_frame_offsets"),
        ({"ge_act_future_indices": (1, 4, 6, 9)}, "ge_act_future_indices"),
    ],
)
def test_production_geometry_rejects_non_planx_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    from qwen35_planx.config import PlanGeometry

    with pytest.raises(ValueError, match=message):
        PlanGeometry(**kwargs)


def test_released_ta_tok_metadata_round_trip_preserves_contracts() -> None:
    from qwen35_planx.config import ReleasedTATokMetadata

    metadata = ReleasedTATokMetadata.example()

    restored = ReleasedTATokMetadata.from_dict(
        json.loads(json.dumps(metadata.to_dict()))
    )

    assert restored == metadata
    assert restored.teacher == "google/siglip2-so400m-patch14-384"
    assert restored.selected_layer == -2
    assert restored.bottleneck_token_num == 729


def test_grounded_planner_metadata_rejects_camera_reordering() -> None:
    from qwen35_planx.config import GroundedPlannerMetadata

    payload = GroundedPlannerMetadata.example().to_dict()
    payload["camera_names"] = ["wrist", "main"]

    with pytest.raises(ValueError, match="camera_names"):
        GroundedPlannerMetadata.from_dict(payload)


def test_grounded_planner_metadata_rejects_incompatible_tokenizer_hash() -> None:
    from qwen35_planx.config import GroundedPlannerMetadata

    expected = GroundedPlannerMetadata.example()

    with pytest.raises(ValueError, match="tokenizer_hash"):
        expected.validate_runtime(tokenizer_hash="different")


def test_json_hash_is_independent_of_key_order() -> None:
    from qwen35_planx.hashing import sha256_json

    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})
