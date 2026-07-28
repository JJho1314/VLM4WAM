from pathlib import Path

import numpy as np
import pytest
import torch

from qwen3_vl_semantic_planner.dinov3_da3_2b.export_libero_episode_siglip2_da3 import (
    artifact_paths,
    da3_depth_images,
    sample_frame_indices,
    siglip_pca_images,
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


def test_siglip_pca_images_use_one_transform_for_all_frames() -> None:
    base = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    features = torch.stack((base, base), dim=0)

    images = siglip_pca_images(features, grid_size=2, output_size=8)

    assert images.shape == (2, 8, 8, 3)
    assert images.dtype == np.uint8
    np.testing.assert_array_equal(images[0], images[1])


def test_da3_depth_images_share_one_episode_color_scale() -> None:
    depth = torch.stack(
        (torch.full((4, 4), 1.0), torch.full((4, 4), 2.0)),
        dim=0,
    )

    images = da3_depth_images(depth)

    assert images.shape == (2, 4, 4, 3)
    assert images.dtype == np.uint8
    assert not np.array_equal(images[0], images[1])


def test_siglip_pca_images_reject_non_square_token_count() -> None:
    with pytest.raises(ValueError, match="grid"):
        siglip_pca_images(torch.zeros(2, 5, 4), grid_size=2, output_size=8)


def test_da3_depth_images_reject_non_positive_depth() -> None:
    with pytest.raises(ValueError, match="positive"):
        da3_depth_images(torch.zeros(2, 4, 4))
