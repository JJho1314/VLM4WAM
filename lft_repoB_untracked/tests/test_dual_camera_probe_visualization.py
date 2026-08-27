from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner"
    / "visualize_dual_camera_probes.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "dual_camera_probe_visualization",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rgb_split_preserves_main_then_wrist_order():
    module = load_module()
    composite = torch.zeros(224, 448, 3, dtype=torch.uint8)
    composite[:, :224, 0] = 255
    composite[:, 224:, 1] = 255

    cameras = module.split_rgb_cameras_224(composite)

    assert tuple(cameras) == ("main", "wrist")
    assert cameras["main"].getpixel((0, 0)) == (255, 0, 0)
    assert cameras["wrist"].getpixel((0, 0)) == (0, 255, 0)


def test_dino_tokens_are_split_before_interpolation():
    module = load_module()
    probe = module.DinoPCAProbe(
        torch.zeros(4),
        torch.eye(4)[:, :3],
        torch.zeros(3),
        torch.ones(3),
    )
    features = torch.zeros(1, 256, 4)
    grid = features.reshape(1, 16, 16, 4)
    grid[:, :, :8, 0] = 1.0
    grid[:, :, 8:, 1] = 1.0

    cameras = module.project_dino_cameras_224(probe, features)

    assert cameras["main"].shape == (1, 3, 224, 224)
    assert cameras["wrist"].shape == (1, 3, 224, 224)
    assert float(cameras["main"][:, 0].mean()) == pytest.approx(1.0)
    assert float(cameras["main"][:, 1].mean()) == pytest.approx(0.0)
    assert float(cameras["wrist"][:, 0].mean()) == pytest.approx(0.0)
    assert float(cameras["wrist"][:, 1].mean()) == pytest.approx(1.0)


def test_depth_alignment_is_independent_per_camera():
    module = load_module()
    relative = torch.zeros(1, 16, 16)
    dense = torch.ones(1, 256, 256)
    dense[..., 128:] = 10.0

    decoded = module.decode_depth_cameras_224(relative, dense)

    assert float(decoded["main"].median()) == pytest.approx(1.0)
    assert float(decoded["wrist"].median()) == pytest.approx(10.0)


def test_camera_split_rejects_odd_width():
    module = load_module()

    with pytest.raises(ValueError, match="width must be even"):
        module.resize_depth_target_cameras_224(torch.ones(1, 32, 31))


def test_dual_camera_output_contract_has_24_separate_pngs(tmp_path):
    module = load_module()
    dino = {
        f"dino_{source}_{camera}_{time}_224": torch.zeros(3, 224, 224)
        for source in ("teacher", "planner")
        for camera in module.CAMERAS
        for time in ("current", "future")
    }
    depth = {
        f"depth_{source}_{camera}_{time}_224": torch.ones(224, 224)
        for source in ("moge", "teacher_probe", "planner_probe")
        for camera in module.CAMERAS
        for time in ("current", "future")
    }

    module.save_dual_camera_sample(
        output_dir=tmp_path,
        current_rgb=torch.zeros(224, 448, 3, dtype=torch.uint8),
        future_rgb=torch.zeros(224, 448, 3, dtype=torch.uint8),
        instruction="pick up the bowl",
        dino_maps=dino,
        depth_maps=depth,
    )

    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 24
    assert not any(
        "combined" in path.name or "query" in path.name
        for path in pngs
    )
    assert (tmp_path / "instruction.txt").read_text() == "pick up the bowl\n"
    assert (tmp_path / "depth_color_ranges.json").is_file()
    for path in pngs:
        with Image.open(path) as image:
            assert image.size == (224, 224)


def test_saved_probes_round_trip(tmp_path):
    module = load_module()
    dino = module.DinoPCAProbe(
        torch.zeros(4),
        torch.eye(4)[:, :3],
        torch.zeros(3),
        torch.ones(3),
    )
    torch.save(
        {"state_dict": dino.state_dict()},
        tmp_path / "dino_pca_probe.pt",
    )
    depth = module.LinearDepthProbe(feature_dim=4, grid_size=16)
    torch.save(
        {
            "state_dict": depth.state_dict(),
            "feature_dim": 4,
            "grid_size": 16,
        },
        tmp_path / "depth_linear_probe.pt",
    )

    loaded_dino, loaded_depth = module.load_saved_probes(
        tmp_path,
        torch.device("cpu"),
    )

    assert torch.equal(loaded_dino.basis, dino.basis)
    assert torch.equal(
        loaded_depth.projection.weight,
        depth.projection.weight,
    )
