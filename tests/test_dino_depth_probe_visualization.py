from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


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
