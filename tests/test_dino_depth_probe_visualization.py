from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner"
    / "train_dino_depth_probe_visualization.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "dino_depth_probe_visualization",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_global_dino_pca_is_deterministic_and_outputs_224():
    module = load_module()
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(6, 256, 12, generator=generator)

    first = module.DinoPCAProbe.fit(features, seed=7)
    second = module.DinoPCAProbe.fit(features, seed=7)
    output = first.project_224(features[:2])

    assert output.shape == (2, 3, 224, 224)
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.basis, second.basis)
    assert torch.equal(first.low, second.low)
    assert torch.equal(first.high, second.high)
    assert torch.isfinite(output).all()
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0


def test_global_dino_pca_does_not_renormalize_each_sample():
    module = load_module()
    generator = torch.Generator().manual_seed(13)
    training = torch.randn(8, 256, 10, generator=generator)
    probe = module.DinoPCAProbe.fit(training, seed=3)
    base = training[:1]
    shifted = base + 0.5

    assert not torch.allclose(
        probe.project_224(base),
        probe.project_224(shifted),
    )


def test_decode_depth_output_is_exactly_224():
    module = load_module()
    generator = torch.Generator().manual_seed(17)
    relative = torch.randn(2, 16, 16, generator=generator)
    target = torch.rand(2, 256, 256, generator=generator).add_(0.1)

    decoded = module.decode_depth_224(relative, target)

    assert decoded.shape == (2, 224, 224)
    assert torch.isfinite(decoded).all()
    assert bool((decoded > 0).all())


def test_sample_outputs_are_separate_and_exactly_224(tmp_path):
    module = load_module()
    paths = module.save_sample_outputs(
        output_dir=tmp_path,
        current_rgb=torch.zeros(224, 448, 3, dtype=torch.uint8),
        future_rgb=torch.zeros(224, 448, 3, dtype=torch.uint8),
        instruction="pick up the bowl",
        dino_maps={
            name: torch.zeros(3, 224, 224)
            for name in module.DINO_OUTPUT_NAMES
        },
        depth_maps={
            name: torch.ones(224, 224)
            for name in module.DEPTH_OUTPUT_NAMES
        },
    )

    assert {path.name for path in paths} == set(module.EXPECTED_SAMPLE_FILES)
    assert (tmp_path / "instruction.txt").read_text() == "pick up the bowl\n"
    assert not any("query" in path.name for path in paths)
    for path in paths:
        if path.suffix == ".png":
            with Image.open(path) as image:
                assert image.size == (224, 224)
