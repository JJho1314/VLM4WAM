# SigLIP2 PCA Upsampling Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a feature-only probe that converts native `16 x 16 x 1024` SigLIP2 tokens into a globally comparable `256 x 256 x 3` PCA visualization, then add that visualization to the sparse LIBERO episode 288 export.

**Architecture:** Put the fixed PCA transform, probe network, loss, validation gate, and checkpoint loader in one focused module. Use a separate training CLI that reads the existing NHWC LIBERO frame cache, freezes two views of the same SigLIP2 checkpoint at 256 and 512 input resolution, and excludes the target episode. Extend the existing exporter with one optional validated-probe path so the original three artifacts remain unchanged and the new export gains `siglip_probe.png`.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, Pillow, Transformers SigLIP2, PyAV, pytest, Slurm on HPC3.

## Global Constraints

- The low-resolution feature contract is the penultimate spatial layer of `siglip2-large-patch16-256`, shaped `[B, 256, 1024]`.
- The PCA teacher contract is the same frozen checkpoint at `512 x 512`, shaped `[B, 1024, 1024]`.
- Probe inference consumes SigLIP2 tokens only; RGB must not be an input.
- PCA mean, components, deterministic signs, and 2nd/98th percentile display limits are fitted on training data and stored in the checkpoint.
- The target LIBERO episode is `libero_10_no_noops_lerobot/episode_000288`; it must not appear in training or validation.
- Both `observation.images.image` and `observation.images.wrist_image` frame-cache files participate in training and validation.
- The current `siglip_pca.png` is preserved and `siglip_probe.png` is added.
- A keeper checkpoint is accepted only if it improves both validation L1 and validation multiscale gradient error over direct low-resolution PCA interpolation.
- The keeper checkpoint is named `siglip2_pca_upsample_probe.pt`; no filename contains `wsa`.
- The new episode export goes to a new directory and contains exactly 120 PNGs and 30 manifest records.

---

### Task 1: Fixed global PCA transform

**Files:**
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_pca_probe.py`
- Create: `tests/test_siglip2_pca_probe.py`

**Interfaces:**
- Produces: `sample_pca_tokens(features: torch.Tensor, max_tokens: int, seed: int) -> torch.Tensor`
- Produces: `fit_fixed_pca(features: torch.Tensor, max_tokens: int, seed: int) -> dict[str, Any]`
- Produces: `project_fixed_pca(features: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor`
- Consumes: floating-point SigLIP2 features whose final dimension is the feature dimension.
- Returns: normalized PCA values in `[0, 1]` with the same leading dimensions and a final dimension of three.

- [ ] **Step 1: Write failing tests for bounded deterministic sampling**

```python
import torch

from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_pca_probe import (
    sample_pca_tokens,
)


def test_sample_pca_tokens_is_bounded_and_deterministic() -> None:
    features = torch.arange(8 * 5, dtype=torch.float32).reshape(2, 4, 5)

    first = sample_pca_tokens(features, max_tokens=5, seed=17)
    second = sample_pca_tokens(features, max_tokens=5, seed=17)

    assert first.shape == (5, 5)
    torch.testing.assert_close(first, second)
```

- [ ] **Step 2: Write failing tests for PCA signs and stored normalization**

```python
from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_pca_probe import (
    fit_fixed_pca,
    project_fixed_pca,
)


def test_fit_fixed_pca_anchors_component_signs() -> None:
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(4, 16, 8, generator=generator)

    state = fit_fixed_pca(features, max_tokens=64, seed=9)
    components = state["components"]
    anchor_columns = components.abs().argmax(dim=1)

    assert state["mean"].shape == (8,)
    assert components.shape == (3, 8)
    assert state["component_sign_rule"] == (
        "largest_absolute_loading_positive"
    )
    assert torch.all(
        components[
            torch.arange(3),
            anchor_columns,
        ] > 0
    )


def test_project_fixed_pca_reuses_global_display_limits() -> None:
    features = torch.tensor(
        [
            [[-2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ]
    )
    state = {
        "mean": torch.zeros(4),
        "components": torch.eye(4)[:3],
        "display_low": torch.tensor([-2.0, -1.0, -1.0]),
        "display_high": torch.tensor([2.0, 1.0, 1.0]),
        "feature_dim": 4,
        "seed": 0,
        "sampled_token_count": 4,
    }

    projected = project_fixed_pca(features, state)

    assert projected.shape == (2, 2, 3)
    assert projected[0, 0, 0].item() == 0.0
    assert projected[0, 1, 0].item() == 1.0
    assert projected[1, 0, 0].item() == 0.25
    assert projected[1, 1, 0].item() == 0.75
```

- [ ] **Step 3: Run the focused tests and verify the module is missing**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py -q
```

Expected: collection fails with `ModuleNotFoundError` for
`siglip2_pca_probe`.

- [ ] **Step 4: Implement bounded token sampling and fixed PCA**

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def sample_pca_tokens(
    features: torch.Tensor,
    *,
    max_tokens: int,
    seed: int,
) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError("PCA features must have a token and feature dimension")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    flat = features.detach().float().reshape(-1, features.shape[-1])
    if flat.shape[0] <= max_tokens:
        return flat
    generator = torch.Generator(device=flat.device).manual_seed(seed)
    indices = torch.randperm(
        flat.shape[0],
        generator=generator,
        device=flat.device,
    )[:max_tokens]
    return flat[indices]


def _anchor_component_signs(components: torch.Tensor) -> torch.Tensor:
    anchor_columns = components.abs().argmax(dim=1)
    anchor_values = components[
        torch.arange(components.shape[0], device=components.device),
        anchor_columns,
    ]
    signs = torch.where(anchor_values < 0, -1.0, 1.0)
    return components * signs[:, None]


def fit_fixed_pca(
    features: torch.Tensor,
    *,
    max_tokens: int = 50_000,
    seed: int = 0,
) -> dict[str, Any]:
    sampled = sample_pca_tokens(
        features,
        max_tokens=max_tokens,
        seed=seed,
    )
    if sampled.shape[0] < 4 or sampled.shape[1] < 3:
        raise ValueError("PCA requires at least four tokens and three features")
    mean = sampled.mean(dim=0)
    centered = sampled - mean
    covariance = centered.T @ centered / max(sampled.shape[0] - 1, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    components = _anchor_component_signs(
        eigenvectors[:, -3:].T.flip(0)
    )
    projected = centered @ components.T
    display_low = torch.quantile(projected, 0.02, dim=0)
    display_high = torch.quantile(projected, 0.98, dim=0)
    if torch.any(display_high <= display_low):
        raise ValueError("PCA display range is degenerate")
    return {
        "mean": mean,
        "components": components,
        "display_low": display_low,
        "display_high": display_high,
        "feature_dim": sampled.shape[1],
        "component_sign_rule": "largest_absolute_loading_positive",
        "max_tokens": max_tokens,
        "seed": seed,
        "sampled_token_count": sampled.shape[0],
    }


def project_fixed_pca(
    features: torch.Tensor,
    state: Mapping[str, Any],
) -> torch.Tensor:
    feature_dim = int(state["feature_dim"])
    if features.shape[-1] != feature_dim:
        raise ValueError(
            f"expected PCA feature dim {feature_dim}, got {features.shape[-1]}"
        )
    values = features.float()
    mean = torch.as_tensor(state["mean"], device=values.device)
    components = torch.as_tensor(
        state["components"],
        device=values.device,
    )
    low = torch.as_tensor(state["display_low"], device=values.device)
    high = torch.as_tensor(state["display_high"], device=values.device)
    projected = (values - mean) @ components.T
    return ((projected - low) / (high - low)).clamp(0, 1)
```

- [ ] **Step 5: Run the Task 1 tests**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py -q
```

Expected: `3 passed`.

- [ ] **Step 6: Commit Task 1**

```bash
git add \
  qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_pca_probe.py \
  tests/test_siglip2_pca_probe.py
git commit -m "feat(viz): add fixed SigLIP2 PCA transform"
```

---

### Task 2: Probe, target, loss, validation gate, and checkpoint contract

**Files:**
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_pca_probe.py`
- Modify: `tests/test_siglip2_pca_probe.py`

**Interfaces:**
- Produces: `SiglipPCAUpsampler(in_dim: int = 1024, hidden_dim: int = 256, grid_size: int = 16, output_size: int = 256)`
- Produces: `pca_target_images(features: torch.Tensor, pca_state: Mapping[str, Any], grid_size: int, output_size: int) -> torch.Tensor`
- Produces: `multiscale_gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor`
- Produces: `validation_metrics(prediction: torch.Tensor, baseline: torch.Tensor, target: torch.Tensor) -> dict[str, float]`
- Produces: `validation_gate_passed(metrics: Mapping[str, float]) -> bool`
- Produces: `load_validated_probe(path: Path, expected_model_name: str, device: torch.device) -> tuple[SiglipPCAUpsampler, dict[str, Any]]`

- [ ] **Step 1: Write failing shape and feature-only interface tests**

```python
import inspect

from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_pca_probe import (
    SiglipPCAUpsampler,
)


def test_siglip_pca_upsampler_is_feature_only_and_dense() -> None:
    probe = SiglipPCAUpsampler(
        in_dim=8,
        hidden_dim=32,
        grid_size=2,
        output_size=32,
    )
    tokens = torch.randn(2, 4, 8)

    output = probe(tokens)

    assert list(inspect.signature(probe.forward).parameters) == ["tokens"]
    assert output.shape == (2, 3, 32, 32)
    assert torch.all((0 <= output) & (output <= 1))
```

- [ ] **Step 2: Write failing tests for target, loss, and validation gate**

```python
from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_pca_probe import (
    multiscale_gradient_loss,
    pca_target_images,
    validation_gate_passed,
    validation_metrics,
)


def test_pca_target_images_projects_then_resizes() -> None:
    features = torch.randn(2, 4, 8)
    state = fit_fixed_pca(features, max_tokens=8, seed=0)

    target = pca_target_images(
        features,
        state,
        grid_size=2,
        output_size=16,
    )

    assert target.shape == (2, 3, 16, 16)
    assert torch.all((0 <= target) & (target <= 1))


def test_multiscale_gradient_loss_is_zero_for_identical_images() -> None:
    image = torch.rand(2, 3, 16, 16)
    assert multiscale_gradient_loss(image, image).item() == 0.0


def test_validation_gate_requires_both_metrics_to_improve() -> None:
    target = torch.zeros(2, 3, 8, 8)
    baseline = torch.ones_like(target)
    probe = torch.full_like(target, 0.25)
    metrics = validation_metrics(probe, baseline, target)

    assert metrics["probe_l1"] < metrics["baseline_l1"]
    assert validation_gate_passed(metrics)
    assert not validation_gate_passed(
        {
            **metrics,
            "probe_gradient": metrics["baseline_gradient"] + 1.0,
        }
    )
```

- [ ] **Step 3: Write failing checkpoint rejection tests**

```python
from pathlib import Path

import pytest

from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_pca_probe import (
    load_validated_probe,
)


def test_load_validated_probe_rejects_failed_gate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rejected.pt"
    torch.save(
        {
            "accepted": False,
            "model_name": "siglip2-large-patch16-256",
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="validation gate"):
        load_validated_probe(
            checkpoint,
            expected_model_name="siglip2-large-patch16-256",
            device=torch.device("cpu"),
        )


def test_load_validated_probe_rejects_incompatible_feature_dim(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "incompatible.pt"
    torch.save(
        {
            "accepted": True,
            "model_name": "siglip2-large-patch16-256",
            "feature_layer": "penultimate_spatial",
            "low_input_size": 256,
            "high_input_size": 512,
            "high_grid_size": 32,
            "state_dict": {},
            "config": {
                "in_dim": 8,
                "hidden_dim": 32,
                "grid_size": 16,
                "output_size": 256,
            },
            "pca_state": {},
            "validation_metrics": {},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="feature contract"):
        load_validated_probe(
            checkpoint,
            expected_model_name="siglip2-large-patch16-256",
            device=torch.device("cpu"),
        )


def test_load_validated_probe_round_trip(tmp_path: Path) -> None:
    checkpoint = tmp_path / "keeper.pt"
    probe = SiglipPCAUpsampler()
    pca_state = {
        "mean": torch.zeros(1024),
        "components": torch.eye(1024)[:3],
        "display_low": torch.zeros(3),
        "display_high": torch.ones(3),
        "feature_dim": 1024,
        "component_sign_rule": "largest_absolute_loading_positive",
        "max_tokens": 64,
        "seed": 5,
        "sampled_token_count": 64,
    }
    torch.save(
        {
            "accepted": True,
            "model_name": "siglip2-large-patch16-256",
            "feature_layer": "penultimate_spatial",
            "low_input_size": 256,
            "high_input_size": 512,
            "high_grid_size": 32,
            "state_dict": probe.state_dict(),
            "config": probe.config(),
            "pca_state": pca_state,
            "validation_metrics": {
                "probe_l1": 0.1,
                "baseline_l1": 0.2,
                "probe_gradient": 0.1,
                "baseline_gradient": 0.2,
            },
        },
        checkpoint,
    )

    loaded, payload = load_validated_probe(
        checkpoint,
        expected_model_name="siglip2-large-patch16-256",
        device=torch.device("cpu"),
    )

    assert isinstance(loaded, SiglipPCAUpsampler)
    assert payload["pca_state"]["sampled_token_count"] == 64
```

- [ ] **Step 4: Run the new tests and verify missing symbols**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py -q
```

Expected: collection fails because `SiglipPCAUpsampler` and the Task 2 helpers
do not exist.

- [ ] **Step 5: Implement the probe and PCA image target**

```python
import torch.nn as nn
import torch.nn.functional as F


class _UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(
            features,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return self.skip(upsampled) + self.refine(upsampled)


class SiglipPCAUpsampler(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 256,
        grid_size: int = 16,
        output_size: int = 256,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.output_size = output_size
        self.input_projection = nn.Conv2d(in_dim, hidden_dim, 1)
        stage_channels = [
            hidden_dim,
            max(hidden_dim // 2, 8),
            max(hidden_dim // 4, 8),
            max(hidden_dim // 8, 8),
        ]
        channel_pairs = zip(
            [hidden_dim, *stage_channels[:-1]],
            stage_channels,
        )
        self.blocks = nn.ModuleList(
            [
                _UpsampleBlock(in_channels, out_channels)
                for in_channels, out_channels in channel_pairs
            ]
        )
        self.head = nn.Conv2d(stage_channels[-1], 3, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.grid_size * self.grid_size
        if tokens.ndim != 3 or tokens.shape[1:] != (
            expected_tokens,
            self.in_dim,
        ):
            raise ValueError(
                f"expected [B,{expected_tokens},{self.in_dim}], "
                f"got {tuple(tokens.shape)}"
            )
        batch = tokens.shape[0]
        features = tokens.transpose(1, 2).reshape(
            batch,
            self.in_dim,
            self.grid_size,
            self.grid_size,
        )
        features = self.input_projection(features)
        for block in self.blocks:
            features = block(features)
        features = F.interpolate(
            features,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(self.head(features))

    def config(self) -> dict[str, int]:
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "grid_size": self.grid_size,
            "output_size": self.output_size,
        }


def pca_target_images(
    features: torch.Tensor,
    pca_state: Mapping[str, Any],
    *,
    grid_size: int,
    output_size: int,
) -> torch.Tensor:
    if features.shape[1] != grid_size * grid_size:
        raise ValueError("feature token count does not match the PCA grid")
    projected = project_fixed_pca(features, pca_state)
    images = projected.reshape(
        features.shape[0],
        grid_size,
        grid_size,
        3,
    ).permute(0, 3, 1, 2)
    return F.interpolate(
        images,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)
```

- [ ] **Step 6: Implement loss and validation metrics**

```python
def multiscale_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    loss = prediction.new_zeros(())
    for scale in scales:
        pred = (
            F.avg_pool2d(prediction, scale)
            if scale > 1
            else prediction
        )
        truth = F.avg_pool2d(target, scale) if scale > 1 else target
        loss = loss + F.l1_loss(
            pred[..., :, 1:] - pred[..., :, :-1],
            truth[..., :, 1:] - truth[..., :, :-1],
        )
        loss = loss + F.l1_loss(
            pred[..., 1:, :] - pred[..., :-1, :],
            truth[..., 1:, :] - truth[..., :-1, :],
        )
    return loss


def validation_metrics(
    prediction: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    return {
        "probe_l1": float(F.l1_loss(prediction, target)),
        "baseline_l1": float(F.l1_loss(baseline, target)),
        "probe_gradient": float(
            multiscale_gradient_loss(prediction, target)
        ),
        "baseline_gradient": float(
            multiscale_gradient_loss(baseline, target)
        ),
    }


def validation_gate_passed(metrics: Mapping[str, float]) -> bool:
    return (
        metrics["probe_l1"] < metrics["baseline_l1"]
        and metrics["probe_gradient"] < metrics["baseline_gradient"]
    )
```

- [ ] **Step 7: Implement strict keeper-checkpoint loading**

```python
from pathlib import Path


def load_validated_probe(
    path: Path,
    *,
    expected_model_name: str,
    device: torch.device,
) -> tuple[SiglipPCAUpsampler, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("accepted", False):
        raise ValueError("probe checkpoint did not pass the validation gate")
    if payload.get("model_name") != expected_model_name:
        raise ValueError(
            "probe checkpoint SigLIP2 model does not match the exporter"
        )
    required = {
        "state_dict",
        "config",
        "pca_state",
        "validation_metrics",
        "feature_layer",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"probe checkpoint is missing: {missing}")
    if payload["feature_layer"] != "penultimate_spatial":
        raise ValueError("probe checkpoint feature layer is incompatible")
    if (
        payload.get("low_input_size") != 256
        or payload.get("high_input_size") != 512
        or payload.get("high_grid_size") != 32
    ):
        raise ValueError("probe checkpoint teacher contract is incompatible")
    config = payload["config"]
    if (
        config.get("in_dim") != 1024
        or config.get("grid_size") != 16
        or config.get("output_size") != 256
    ):
        raise ValueError("probe checkpoint feature contract is incompatible")
    required_pca = {
        "mean",
        "components",
        "display_low",
        "display_high",
        "feature_dim",
        "component_sign_rule",
        "max_tokens",
    }
    missing_pca = sorted(required_pca.difference(payload["pca_state"]))
    if missing_pca:
        raise ValueError(f"probe PCA state is missing: {missing_pca}")
    if (
        payload["pca_state"]["feature_dim"] != 1024
        or payload["pca_state"]["component_sign_rule"]
        != "largest_absolute_loading_positive"
    ):
        raise ValueError("probe PCA feature contract is incompatible")
    probe = SiglipPCAUpsampler(**config).to(device).eval()
    probe.load_state_dict(payload["state_dict"])
    probe.requires_grad_(False)
    return probe, payload
```

- [ ] **Step 8: Run all probe tests**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py -q
```

Expected: `10 passed`.

- [ ] **Step 9: Commit Task 2**

```bash
git add \
  qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_pca_probe.py \
  tests/test_siglip2_pca_probe.py
git commit -m "feat(viz): add SigLIP2 PCA upsampling probe"
```

---

### Task 3: Episode-safe frame-cache dataset and training CLI

**Files:**
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/train_siglip2_pca_probe.py`
- Modify: `tests/test_siglip2_pca_probe.py`

**Interfaces:**
- Produces: `discover_episode_files(cache_root: Path) -> list[Path]`
- Produces: `split_episode_files(files: Sequence[Path], cache_root: Path, validation_modulus: int = 10) -> tuple[list[Path], list[Path]]`
- Produces: `CachedFrameDataset(files: Sequence[Path], virtual_length: int, seed: int)`
- Produces CLI flags: `--frame-cache-dir`, `--siglip2-model-dir`, `--output-dir`, `--steps`, `--batch-size`, `--pca-batches`, `--validation-batches`, `--seed`, `--device`.
- Consumes the HPC3 cache layout `suite/videos/chunk-000/camera/episode_NNNNNN.npy`.
- Returns `[3, H, W]` floating-point frames in `[0, 1]` from NHWC or NCHW cache arrays.

- [ ] **Step 1: Write failing split tests with both cameras grouped**

```python
from qwen3_vl_semantic_planner.dinov3_da3_2b.train_siglip2_pca_probe import (
    split_episode_files,
)


def test_split_episode_files_excludes_only_target_episode(
    tmp_path: Path,
) -> None:
    relative = [
        "libero_10_no_noops_lerobot/videos/chunk-000/"
        "observation.images.image/episode_000288.npy",
        "libero_10_no_noops_lerobot/videos/chunk-000/"
        "observation.images.wrist_image/episode_000288.npy",
        "libero_goal_no_noops_lerobot/videos/chunk-000/"
        "observation.images.image/episode_000288.npy",
        "libero_goal_no_noops_lerobot/videos/chunk-000/"
        "observation.images.wrist_image/episode_000288.npy",
        "libero_10_no_noops_lerobot/videos/chunk-000/"
        "observation.images.image/episode_000100.npy",
        "libero_10_no_noops_lerobot/videos/chunk-000/"
        "observation.images.wrist_image/episode_000100.npy",
    ]
    files = [tmp_path / value for value in relative]

    train, validation = split_episode_files(
        files,
        cache_root=tmp_path,
        validation_modulus=2,
    )
    selected = train + validation

    assert all(
        "libero_10_no_noops_lerobot" not in str(path)
        or "episode_000288.npy" not in path.name
        for path in selected
    )
    assert sum(path.name == "episode_000288.npy" for path in selected) == 2
    for suite_episode in {
        (
            path.parts[-5],
            path.name,
        )
        for path in selected
    }:
        members = [
            path
            for path in selected
            if (path.parts[-5], path.name) == suite_episode
        ]
        assert all(path in train for path in members) or all(
            path in validation for path in members
        )
```

- [ ] **Step 2: Write failing cache-layout conversion test**

```python
import numpy as np

from qwen3_vl_semantic_planner.dinov3_da3_2b.train_siglip2_pca_probe import (
    CachedFrameDataset,
)


def test_cached_frame_dataset_reads_nhwc_uint8(tmp_path: Path) -> None:
    path = tmp_path / "episode_000001.npy"
    np.save(path, np.full((2, 12, 10, 3), 128, dtype=np.uint8))
    dataset = CachedFrameDataset([path], virtual_length=2, seed=7)

    frame = dataset[0]

    assert frame.shape == (3, 12, 10)
    assert frame.dtype == torch.float32
    assert frame.mean().item() == pytest.approx(128 / 255)
```

- [ ] **Step 3: Write failing CLI-default test**

```python
from qwen3_vl_semantic_planner.dinov3_da3_2b.train_siglip2_pca_probe import (
    build_parser as build_probe_parser,
)


def test_probe_training_parser_uses_approved_recipe() -> None:
    args = build_probe_parser().parse_args(
        [
            "--frame-cache-dir",
            "/data/cache",
            "--siglip2-model-dir",
            "/models/siglip2-large-patch16-256",
            "--output-dir",
            "/outputs/probe",
        ]
    )

    assert args.steps == 5000
    assert args.pca_max_tokens == 50_000
    assert args.pca_batches > 0
    assert args.validation_batches > 0
    assert args.seed == 0
```

- [ ] **Step 4: Run the tests and verify the training module is missing**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py -q
```

Expected: collection fails with `ModuleNotFoundError` for
`train_siglip2_pca_probe`.

- [ ] **Step 5: Implement deterministic episode discovery and split**

```python
import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

TARGET_SUITE = "libero_10_no_noops_lerobot"
TARGET_EPISODE = 288
EPISODE_PATTERN = re.compile(r"episode_(\d{6})\.npy$")


def discover_episode_files(cache_root: Path) -> list[Path]:
    files = sorted(cache_root.glob("**/observation.images.*/episode_*.npy"))
    if not files:
        raise FileNotFoundError(
            f"no LIBERO episode frame caches under {cache_root}"
        )
    return files


def _episode_identity(path: Path, cache_root: Path) -> tuple[str, int]:
    relative = path.relative_to(cache_root)
    match = EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid episode cache name: {path}")
    return relative.parts[0], int(match.group(1))


def split_episode_files(
    files: Sequence[Path],
    *,
    cache_root: Path,
    validation_modulus: int = 10,
) -> tuple[list[Path], list[Path]]:
    if validation_modulus <= 1:
        raise ValueError("validation_modulus must exceed one")
    train: list[Path] = []
    validation: list[Path] = []
    for path in sorted(files):
        suite, episode = _episode_identity(path, cache_root)
        if suite == TARGET_SUITE and episode == TARGET_EPISODE:
            continue
        key = f"{suite}/episode_{episode:06d}".encode()
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        destination = (
            validation
            if bucket % validation_modulus == 0
            else train
        )
        destination.append(path)
    if not train or not validation:
        raise RuntimeError("episode split produced an empty partition")
    return train, validation
```

- [ ] **Step 6: Implement NHWC/NCHW cache sampling**

```python
import random

import numpy as np


class CachedFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        *,
        virtual_length: int,
        seed: int,
    ) -> None:
        if not files:
            raise ValueError("CachedFrameDataset requires episode files")
        self.files = tuple(files)
        self.virtual_length = virtual_length
        self.seed = seed
        self._memmaps: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return self.virtual_length

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = random.Random(self.seed + index * 2_654_435_761)
        path = self.files[generator.randrange(len(self.files))]
        array = self._memmaps.get(path)
        if array is None:
            array = np.load(path, mmap_mode="r")
            self._memmaps[path] = array
        frame = np.ascontiguousarray(
            array[generator.randrange(array.shape[0])]
        )
        if frame.ndim != 3:
            raise ValueError(f"expected an image frame, got {frame.shape}")
        if frame.shape[-1] == 3:
            frame = np.moveaxis(frame, -1, 0)
        elif frame.shape[0] != 3:
            raise ValueError(f"cannot infer cache layout from {frame.shape}")
        return torch.from_numpy(frame).float().div_(255.0)
```

- [ ] **Step 7: Implement the parser and frozen teacher setup**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-cache-dir", type=Path, required=True)
    parser.add_argument("--siglip2-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--pca-batches", type=int, default=25)
    parser.add_argument("--pca-max-tokens", type=int, default=50_000)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def make_teachers(
    model_dir: Path,
    device: torch.device,
) -> tuple[Siglip2TargetEncoder, Siglip2TargetEncoder]:
    low = Siglip2TargetEncoder(
        model_dir=model_dir,
        input_size=256,
        grid_size=16,
        device=device,
    )
    high = Siglip2TargetEncoder(
        model_dir=model_dir,
        input_size=512,
        grid_size=0,
        device=device,
    )
    return low, high
```

- [ ] **Step 8: Implement PCA fitting, training, and validation**

Use one deterministic loader per stage. During PCA fitting, concatenate at
most `pca_max_tokens` high-resolution teacher tokens before calling
`fit_fixed_pca`. During training, use:

```python
with torch.inference_mode():
    low_tokens = low_teacher._patch_tokens(low_teacher._prep(frames)).float()
    high_tokens = high_teacher._patch_tokens(
        high_teacher._prep(frames)
    ).float()
    target = pca_target_images(
        high_tokens,
        pca_state,
        grid_size=32,
        output_size=256,
    )

prediction = probe(low_tokens)
pixel_loss = F.l1_loss(prediction, target)
edge_loss = multiscale_gradient_loss(prediction, target)
loss = pixel_loss + 0.25 * edge_loss
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
scheduler.step()
```

For validation, accumulate sums over exactly `validation_batches`. Build the
baseline with the stored PCA:

```python
baseline = pca_target_images(
    low_tokens,
    pca_state,
    grid_size=16,
    output_size=256,
)
batch_metrics = validation_metrics(
    prediction=probe(low_tokens),
    baseline=baseline,
    target=target,
)
```

Average each metric by batch count, call `validation_gate_passed`, and emit one
JSON line containing the aggregate metrics and `accepted`.

- [ ] **Step 9: Implement accepted and rejected checkpoint naming**

```python
payload = {
    "accepted": accepted,
    "model_name": args.siglip2_model_dir.name,
    "feature_layer": "penultimate_spatial",
    "low_input_size": 256,
    "high_input_size": 512,
    "high_grid_size": 32,
    "state_dict": probe.state_dict(),
    "config": probe.config(),
    "pca_state": pca_state,
    "validation_metrics": aggregate_metrics,
    "split": {
        "target_exclusion": (
            "libero_10_no_noops_lerobot/episode_000288"
        ),
        "train_files": len(train_files),
        "validation_files": len(validation_files),
        "train_relative_paths": [
            str(path.relative_to(args.frame_cache_dir))
            for path in train_files
        ],
        "validation_relative_paths": [
            str(path.relative_to(args.frame_cache_dir))
            for path in validation_files
        ],
        "validation_modulus": 10,
    },
    "training": {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    },
}
filename = (
    "siglip2_pca_upsample_probe.pt"
    if accepted
    else "siglip2_pca_upsample_probe_rejected.pt"
)
torch.save(payload, args.output_dir / filename)
if not accepted:
    raise SystemExit(2)
```

- [ ] **Step 10: Run probe and exporter regression tests**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py \
  tests/test_export_libero_episode_siglip2_da3.py \
  tests/test_ge_act_dual_camera_planner.py -q
```

Expected: all tests pass.

- [ ] **Step 11: Commit Task 3**

```bash
git add \
  qwen3_vl_semantic_planner/dinov3_da3_2b/train_siglip2_pca_probe.py \
  tests/test_siglip2_pca_probe.py
git commit -m "feat(viz): train SigLIP2 PCA upsampling probe"
```

---

### Task 4: Optional probe integration in the episode exporter

**Files:**
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py`
- Modify: `tests/test_export_libero_episode_siglip2_da3.py`

**Interfaces:**
- Changes: `artifact_paths(output_dir: Path, camera: str, frame_index: int, include_siglip_probe: bool = False) -> dict[str, Path]`
- Changes: `write_export(..., siglip_probe_rgb: np.ndarray | None = None) -> list[dict[str, Any]]`
- Adds CLI flag: `--siglip-pca-probe PATH`.
- Consumes: a keeper checkpoint loaded through `load_validated_probe`.
- Produces: `siglip_probe.png` only when the optional checkpoint is supplied.

- [ ] **Step 1: Write failing optional-path test**

```python
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
```

- [ ] **Step 2: Write failing four-file export test**

```python
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
    assert (
        tmp_path / "main/frame_000000/siglip_pca.png"
    ).is_file()
    assert (
        tmp_path / "main/frame_000000/siglip_probe.png"
    ).is_file()
```

- [ ] **Step 3: Extend the parser test**

Add `--siglip-pca-probe /probes/siglip.pt` to the existing parser test and
assert:

```python
assert args.siglip_pca_probe == Path("/probes/siglip.pt")
```

- [ ] **Step 4: Run exporter tests and verify failures**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_export_libero_episode_siglip2_da3.py -q
```

Expected: the new tests fail because the optional argument and artifact do not
exist.

- [ ] **Step 5: Add optional paths and writes**

```python
def artifact_paths(
    output_dir: Path,
    camera: str,
    frame_index: int,
    *,
    include_siglip_probe: bool = False,
) -> dict[str, Path]:
    frame_dir = output_dir / camera / f"frame_{frame_index:06d}"
    paths = {
        "rgb": frame_dir / "rgb.png",
        "siglip_pca": frame_dir / "siglip_pca.png",
        "da3_depth": frame_dir / "da3_depth.png",
    }
    if include_siglip_probe:
        paths["siglip_probe"] = frame_dir / "siglip_probe.png"
    return paths
```

In `write_export`, request the optional path when
`siglip_probe_rgb is not None`, write the corresponding flattened camera/frame
index, and include it in the record's `files` mapping. Keep the original
three-image behavior byte-for-byte when `siglip_probe_rgb` is `None`.

- [ ] **Step 6: Load and run the validated probe**

Add:

```python
parser.add_argument("--siglip-pca-probe", type=Path, default=None)
```

After native SigLIP2 features have been encoded and before DA3 is loaded:

```python
siglip_probe_rgb = None
probe_payload = None
if args.siglip_pca_probe is not None:
    from siglip2_pca_probe import load_validated_probe

    probe, probe_payload = load_validated_probe(
        _required_path(args.siglip_pca_probe, "SigLIP2 PCA probe"),
        expected_model_name=siglip2_model_dir.name,
        device=device,
    )
    with torch.inference_mode():
        dense = probe(siglip_features.to(device).float())
    siglip_probe_rgb = (
        dense.mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    del probe
    torch.cuda.empty_cache()
```

Pass `siglip_probe_rgb` to `write_export`. Set expected modalities to four when
the checkpoint is present. Add the probe path and validation metrics to the
manifest:

```python
if probe_payload is not None:
    manifest["models"]["siglip2_pca_probe"] = str(
        args.siglip_pca_probe
    )
    manifest["siglip2_probe_validation"] = probe_payload[
        "validation_metrics"
    ]
```

- [ ] **Step 7: Run focused and related tests**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py \
  tests/test_export_libero_episode_siglip2_da3.py \
  tests/test_ge_act_dual_camera_planner.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add \
  qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py \
  tests/test_export_libero_episode_siglip2_da3.py
git commit -m "feat(viz): export dense SigLIP2 probe images"
```

---

### Task 5: Reproducible HPC3 launcher and operator documentation

**Files:**
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/sbatch_train_siglip2_pca_probe_hpc3.sh`
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/README_probes_viz.md`
- Modify: `tests/test_siglip2_pca_probe.py`

**Interfaces:**
- Launcher accepts environment overrides `RUN_KIND`, `STEPS`, `BATCH_SIZE`, `OUTPUT_DIR`, `FRAME_CACHE_DIR`, `SIGLIP2_MODEL_DIR`, `PYTHON`, and `REPO_ROOT`.
- `RUN_KIND=smoke` forces 2 steps, 1 PCA batch, and 1 validation batch.
- Formal defaults are 5,000 steps, batch size 8, 25 PCA batches, and 50 validation batches.

- [ ] **Step 1: Write a failing launcher-contract test**

```python
def test_hpc3_launcher_records_probe_recipe() -> None:
    launcher = (
        Path(__file__).resolve().parents[1]
        / "qwen3_vl_semantic_planner/dinov3_da3_2b/"
        "sbatch_train_siglip2_pca_probe_hpc3.sh"
    ).read_text()

    assert "#SBATCH --partition=acd_u" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "libero_fastwam_frame_cache_160" in launcher
    assert "siglip2-large-patch16-256" in launcher
    assert "RUN_KIND" in launcher
    assert "--pca-batches" in launcher
    assert "--validation-batches" in launcher
    assert "siglip2_pca_upsample_probe.pt" not in launcher
```

- [ ] **Step 2: Run the test and verify the launcher is missing**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py::test_hpc3_launcher_records_probe_recipe -q
```

Expected: failure with `FileNotFoundError`.

- [ ] **Step 3: Add the Slurm launcher**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=sig2_pca_probe
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jhe724/junjie/logs/sig2-pca-probe-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af}"
PYTHON="${PYTHON:-/data/user/jhe724/.venvs/vlm4wam_joint/bin/python}"
FRAME_CACHE_DIR="${FRAME_CACHE_DIR:-/data/user/jhe724/workspace/data/libero_fastwam_frame_cache_160}"
SIGLIP2_MODEL_DIR="${SIGLIP2_MODEL_DIR:-/data/user/jhe724/junjie/weights/siglip2-large-patch16-256}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/user/jhe724/junjie/probes_2b/siglip2_pca_upsample}"
RUN_KIND="${RUN_KIND:-formal}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ "$RUN_KIND" == "smoke" ]]; then
    STEPS="${STEPS:-2}"
    PCA_BATCHES="${PCA_BATCHES:-1}"
    VALIDATION_BATCHES="${VALIDATION_BATCHES:-1}"
else
    STEPS="${STEPS:-5000}"
    PCA_BATCHES="${PCA_BATCHES:-25}"
    VALIDATION_BATCHES="${VALIDATION_BATCHES:-50}"
fi

for path in \
    "$PYTHON" \
    "$FRAME_CACHE_DIR" \
    "$SIGLIP2_MODEL_DIR/config.json"; do
    [[ -e "$path" ]] || {
        echo "missing required path: $path" >&2
        exit 2
    }
done

mkdir -p "$OUTPUT_DIR" /data/user/jhe724/junjie/logs
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12

exec "$PYTHON" \
    "$REPO_ROOT/qwen3_vl_semantic_planner/dinov3_da3_2b/train_siglip2_pca_probe.py" \
    --frame-cache-dir "$FRAME_CACHE_DIR" \
    --siglip2-model-dir "$SIGLIP2_MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --pca-batches "$PCA_BATCHES" \
    --validation-batches "$VALIDATION_BATCHES" \
    --device cuda
```

- [ ] **Step 4: Document training, gate, and episode export**

Append a `SigLIP2 PCA upsampling probe` section to
`README_probes_viz.md` that records:

```text
Input: penultimate 16x16x1024 SigLIP2-large tokens.
Target: the same frozen model at 512 input, fixed global PCA, 32x32 -> 256x256.
Keeper: siglip2_pca_upsample_probe.pt.
Rejected run: siglip2_pca_upsample_probe_rejected.pt and exit code 2.
The probe is feature-only and does not accept RGB.
```

Include the exact `sbatch` submission command and the exporter
`--siglip-pca-probe` flag.

- [ ] **Step 5: Run tests and shell syntax validation**

Run:

```bash
bash -n \
  qwen3_vl_semantic_planner/dinov3_da3_2b/sbatch_train_siglip2_pca_probe_hpc3.sh
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py \
  tests/test_export_libero_episode_siglip2_da3.py \
  tests/test_ge_act_dual_camera_planner.py -q
```

Expected: shell syntax exits zero and all tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add \
  qwen3_vl_semantic_planner/dinov3_da3_2b/sbatch_train_siglip2_pca_probe_hpc3.sh \
  qwen3_vl_semantic_planner/dinov3_da3_2b/README_probes_viz.md \
  tests/test_siglip2_pca_probe.py
git commit -m "feat(viz): launch SigLIP2 PCA probe training"
```

---

### Task 6: HPC3 smoke run, full training, and episode 288 export

**Files:**
- Runtime checkpoint: `/data/user/jhe724/junjie/probes_2b/siglip2_pca_upsample/siglip2_pca_upsample_probe.pt`
- Runtime remote output: `/data/user/jhe724/junjie/outputs/libero_episode_000288_siglip2_da3_stride16_probe/`
- Runtime local output: `/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/libero_episode_000288_siglip2_da3_stride16_probe/`

**Interfaces:**
- Consumes the committed training CLI, launcher, probe module, and exporter.
- Produces one accepted keeper checkpoint and one new 120-PNG episode export.

- [ ] **Step 1: Run fresh local verification before synchronization**

Run:

```bash
bash -n \
  qwen3_vl_semantic_planner/dinov3_da3_2b/sbatch_train_siglip2_pca_probe_hpc3.sh
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m py_compile \
  qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_pca_probe.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/train_siglip2_pca_probe.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py \
  tests/test_export_libero_episode_siglip2_da3.py \
  tests/test_ge_act_dual_camera_planner.py -q
```

Expected: all commands exit zero.

- [ ] **Step 2: Preflight exact remote paths**

Run:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 hpc3 '
for path in \
  /data/user/jhe724/.venvs/vlm4wam_joint/bin/python \
  /data/user/jhe724/workspace/data/libero_fastwam_frame_cache_160 \
  /data/user/jhe724/junjie/weights/siglip2-large-patch16-256/config.json \
  /data/user/jhe724/junjie/vlm4wam_joint_assets/DA3-LARGE-1.1 \
  /data/user/jhe724/junjie/vlm4wam_joint_assets/Depth-Anything-3; do
  test -e "$path" || exit 2
done
echo HPC3_PREFLIGHT_OK
'
```

Expected: `HPC3_PREFLIGHT_OK`.

- [ ] **Step 3: Synchronize exact source files and compare hashes**

Run:

```bash
REMOTE_REPO=/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af
rsync -av \
  qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_pca_probe.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/train_siglip2_pca_probe.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/sbatch_train_siglip2_pca_probe_hpc3.sh \
  "hpc3:$REMOTE_REPO/qwen3_vl_semantic_planner/dinov3_da3_2b/"
sha256sum \
  qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_pca_probe.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/train_siglip2_pca_probe.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py
ssh hpc3 "sha256sum \
  $REMOTE_REPO/qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_pca_probe.py \
  $REMOTE_REPO/qwen3_vl_semantic_planner/dinov3_da3_2b/train_siglip2_pca_probe.py \
  $REMOTE_REPO/qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py"
```

Expected: every local/remote hash pair matches.

- [ ] **Step 4: Submit and verify a two-step smoke job**

Run:

```bash
ssh hpc3 '
cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af
RUN_KIND=smoke \
OUTPUT_DIR=/data/user/jhe724/junjie/probes_2b/siglip2_pca_upsample_smoke \
sbatch qwen3_vl_semantic_planner/dinov3_da3_2b/sbatch_train_siglip2_pca_probe_hpc3.sh
'
```

Capture the job ID, monitor `squeue`, `sacct`, and its log. The expected result
is either an accepted or rejected smoke checkpoint; the two-step smoke run is
not required to pass the validation gate. It must load both teachers, complete
forward/backward, run validation, and write a checkpoint without a traceback.

- [ ] **Step 5: Submit the formal 5,000-step job**

After smoke completion, run:

```bash
ssh hpc3 '
cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af
RUN_KIND=formal \
OUTPUT_DIR=/data/user/jhe724/junjie/probes_2b/siglip2_pca_upsample \
sbatch qwen3_vl_semantic_planner/dinov3_da3_2b/sbatch_train_siglip2_pca_probe_hpc3.sh
'
```

Monitor at intervals no longer than 60 seconds while actively working. Require
`sacct` state `COMPLETED` with exit code `0:0`.

- [ ] **Step 6: Verify the validation gate and checkpoint contract**

Run:

```bash
ssh hpc3 '
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python - <<"PY"
from pathlib import Path
import torch

path = Path(
    "/data/user/jhe724/junjie/probes_2b/"
    "siglip2_pca_upsample/siglip2_pca_upsample_probe.pt"
)
payload = torch.load(path, map_location="cpu", weights_only=False)
assert payload["accepted"] is True
metrics = payload["validation_metrics"]
assert metrics["probe_l1"] < metrics["baseline_l1"]
assert metrics["probe_gradient"] < metrics["baseline_gradient"]
assert payload["split"]["target_exclusion"] == (
    "libero_10_no_noops_lerobot/episode_000288"
)
print("PROBE_KEEPER_OK", metrics)
PY
'
```

Expected: `PROBE_KEEPER_OK` and both strict metric improvements.

- [ ] **Step 7: Submit the new episode export**

Run the exporter as a single-GPU debug job with:

```text
--data-root /data/user/jhe724/workspace/data/libero_fastwam
--suite libero_10_no_noops_lerobot
--episode-index 288
--stride 16
--siglip2-model-dir /data/user/jhe724/junjie/weights/siglip2-large-patch16-256
--siglip-pca-probe /data/user/jhe724/junjie/probes_2b/siglip2_pca_upsample/siglip2_pca_upsample_probe.pt
--da3-ckpt-dir /data/user/jhe724/junjie/vlm4wam_joint_assets/DA3-LARGE-1.1
--da3-code-root /data/user/jhe724/junjie/vlm4wam_joint_assets/Depth-Anything-3
--output-dir /data/user/jhe724/junjie/outputs/libero_episode_000288_siglip2_da3_stride16_probe
--batch-size 8
--device cuda
```

Require `sacct` state `COMPLETED`, exit code `0:0`, and exporter log status
`done`.

- [ ] **Step 8: Synchronize the episode result locally**

Run:

```bash
mkdir -p /data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs
rsync -av \
  hpc3:/data/user/jhe724/junjie/outputs/libero_episode_000288_siglip2_da3_stride16_probe/ \
  /data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/libero_episode_000288_siglip2_da3_stride16_probe/
```

- [ ] **Step 9: Verify the manifest and all 120 PNG files**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python - <<'PY'
import json
from pathlib import Path
from PIL import Image

root = Path(
    "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/"
    "libero_episode_000288_siglip2_da3_stride16_probe"
)
manifest = json.loads((root / "manifest.json").read_text())
expected = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 223]
assert manifest["frame_indices"] == expected
assert len(manifest["records"]) == 30
assert "siglip2_pca_probe" in manifest["models"]
pngs = list(root.rglob("*.png"))
assert len(pngs) == 120
for record in manifest["records"]:
    assert set(record["files"]) == {
        "rgb",
        "siglip_pca",
        "siglip_probe",
        "da3_depth",
    }
    for relative in record["files"].values():
        with Image.open(root / relative) as image:
            image.verify()
print("PROBE_EXPORT_OK frames=15 cameras=2 records=30 pngs=120")
PY
```

Expected:
`PROBE_EXPORT_OK frames=15 cameras=2 records=30 pngs=120`.

- [ ] **Step 10: Inspect independent images**

Open these individual files, not a composite panel:

```text
main/frame_000000/siglip_pca.png
main/frame_000000/siglip_probe.png
main/frame_000112/siglip_pca.png
main/frame_000112/siglip_probe.png
main/frame_000223/siglip_probe.png
wrist/frame_000000/siglip_probe.png
wrist/frame_000112/siglip_probe.png
wrist/frame_000223/siglip_probe.png
```

Confirm that the probe files are readable, use the same global color
convention, and show sharper spatial transitions than the corresponding
directly interpolated PCA maps without copying the RGB appearance.

- [ ] **Step 11: Run final regression tests and inspect git state**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_pca_probe.py \
  tests/test_export_libero_episode_siglip2_da3.py \
  tests/test_ge_act_dual_camera_planner.py -q
git status --short
git log --oneline -8
```

Expected: all related tests pass and the worktree has no uncommitted source
changes.
