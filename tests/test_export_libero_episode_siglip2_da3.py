import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from qwen3_vl_semantic_planner.dinov3_da3_2b.export_libero_episode_siglip2_da3 import (
    artifact_paths,
    build_parser,
    da3_depth_images,
    decode_episode_frames,
    load_episode_record,
    sample_frame_indices,
    siglip_pca_images,
    write_export,
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


def test_artifact_paths_optionally_adds_probe_image(tmp_path: Path) -> None:
    paths = artifact_paths(
        tmp_path,
        "main",
        16,
        include_siglip_probe=True,
    )

    assert paths["siglip_probe"] == (
        tmp_path / "main/frame_000016/siglip_probe.png"
    )
    assert set(paths) == {
        "rgb",
        "siglip_pca",
        "siglip_probe",
        "da3_depth",
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


def test_load_episode_record_returns_exact_episode(tmp_path: Path) -> None:
    meta = tmp_path / "suite/meta"
    meta.mkdir(parents=True)
    records = [
        {"episode_index": 7, "tasks": ["task seven"], "length": 9},
        {"episode_index": 8, "tasks": ["task eight"], "length": 10},
    ]
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert load_episode_record(tmp_path, "suite", 8) == records[1]


def test_decode_episode_frames_reads_exact_requested_indices(
    tmp_path: Path,
) -> None:
    import av

    video_path = tmp_path / "episode.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width = 8
        stream.height = 8
        stream.pix_fmt = "yuv420p"
        for value in (0, 50, 100, 150):
            array = np.full((8, 8, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    decoded = decode_episode_frames(video_path, [0, 2, 3])

    assert decoded.shape == (3, 8, 8, 3)
    assert decoded[0].mean() < decoded[1].mean() < decoded[2].mean()


def test_decode_episode_frames_rejects_missing_requested_indices(
    tmp_path: Path,
) -> None:
    import av

    video_path = tmp_path / "short.mp4"
    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width = 8
        stream.height = 8
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(
            np.zeros((8, 8, 3), dtype=np.uint8),
            format="rgb24",
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    with pytest.raises(RuntimeError, match="missing requested frames"):
        decode_episode_frames(video_path, [0, 2])


def test_write_export_creates_three_independent_pngs_per_camera_frame(
    tmp_path: Path,
) -> None:
    frames = np.zeros((2, 2, 4, 4, 3), dtype=np.uint8)
    siglip = np.full((4, 8, 8, 3), 64, dtype=np.uint8)
    depth = np.full((4, 4, 4, 3), 128, dtype=np.uint8)

    records = write_export(
        tmp_path,
        frames=frames,
        siglip_rgb=siglip,
        depth_rgb=depth,
        camera_names=("main", "wrist"),
        frame_indices=[0, 8],
        fps=20.0,
    )

    assert len(records) == 4
    assert len(list(tmp_path.rglob("*.png"))) == 12
    assert Image.open(tmp_path / "main/frame_000000/rgb.png").size == (4, 4)
    assert Image.open(
        tmp_path / "main/frame_000000/siglip_pca.png"
    ).size == (8, 8)
    assert records[-1]["timestamp_seconds"] == pytest.approx(0.4)


def test_write_export_adds_probe_without_replacing_pca(
    tmp_path: Path,
) -> None:
    frames = np.zeros((2, 1, 4, 4, 3), dtype=np.uint8)
    siglip = np.full((2, 8, 8, 3), 64, dtype=np.uint8)
    probe = np.full((2, 8, 8, 3), 96, dtype=np.uint8)
    depth = np.full((2, 4, 4, 3), 128, dtype=np.uint8)

    records = write_export(
        tmp_path,
        frames=frames,
        siglip_rgb=siglip,
        siglip_probe_rgb=probe,
        depth_rgb=depth,
        camera_names=("main", "wrist"),
        frame_indices=[0],
        fps=20.0,
    )

    assert len(records) == 2
    assert len(list(tmp_path.rglob("*.png"))) == 8
    assert (tmp_path / "main/frame_000000/siglip_pca.png").is_file()
    assert (tmp_path / "main/frame_000000/siglip_probe.png").is_file()


def test_build_parser_exposes_full_episode_export_contract() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--data-root",
            "/data/libero",
            "--siglip2-model-dir",
            "/models/siglip2",
            "--siglip-pca-probe",
            "/probes/siglip.pt",
            "--da3-ckpt-dir",
            "/models/da3",
            "--da3-code-root",
            "/code/da3",
            "--output-dir",
            "/tmp/export",
        ]
    )

    assert args.suite == "libero_10_no_noops_lerobot"
    assert args.episode_index == 288
    assert args.stride == 16
    assert args.batch_size == 8
    assert args.device == "cuda"
    assert args.siglip_pca_probe == Path("/probes/siglip.pt")
