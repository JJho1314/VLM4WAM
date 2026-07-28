from pathlib import Path

import pytest

from qwen3_vl_semantic_planner.dinov3_da3_2b.export_libero_episode_siglip2_da3 import (
    artifact_paths,
    sample_frame_indices,
)


def test_sample_frame_indices_covers_episode_end() -> None:
    assert sample_frame_indices(224, 16) == [
        0,
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        176,
        192,
        208,
        223,
    ]


@pytest.mark.parametrize(
    ("num_frames", "stride"),
    [(0, 16), (224, 0), (224, -1)],
)
def test_sample_frame_indices_rejects_invalid_inputs(
    num_frames: int,
    stride: int,
) -> None:
    with pytest.raises(ValueError):
        sample_frame_indices(num_frames, stride)


def test_artifact_paths_separates_camera_frame_and_modalities(
    tmp_path: Path,
) -> None:
    assert artifact_paths(tmp_path, "wrist", 32) == {
        "rgb": tmp_path / "wrist/frame_000032/rgb.png",
        "siglip_pca": tmp_path / "wrist/frame_000032/siglip_pca.png",
        "da3_depth": tmp_path / "wrist/frame_000032/da3_depth.png",
    }
