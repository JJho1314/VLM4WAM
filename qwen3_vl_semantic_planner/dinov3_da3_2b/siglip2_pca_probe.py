from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    components = _anchor_component_signs(eigenvectors[:, -3:].T.flip(0))
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


def load_validated_probe(
    path: Path,
    *,
    expected_model_name: str,
    device: torch.device,
) -> tuple[SiglipPCAUpsampler, dict[str, Any]]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise ValueError("probe checkpoint root must be a mapping")
    payload = dict(loaded)
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
    if not validation_gate_passed(payload["validation_metrics"]):
        raise ValueError("probe checkpoint did not pass the validation gate")
    required_pca = {
        "mean",
        "components",
        "display_low",
        "display_high",
        "feature_dim",
        "component_sign_rule",
        "max_tokens",
        "seed",
        "sampled_token_count",
    }
    pca_state = payload["pca_state"]
    missing_pca = sorted(required_pca.difference(pca_state))
    if missing_pca:
        raise ValueError(f"probe PCA state is missing: {missing_pca}")
    if (
        pca_state["feature_dim"] != 1024
        or pca_state["component_sign_rule"]
        != "largest_absolute_loading_positive"
    ):
        raise ValueError("probe PCA feature contract is incompatible")
    if (
        type(pca_state["seed"]) is not int
        or type(pca_state["max_tokens"]) is not int
        or type(pca_state["sampled_token_count"]) is not int
        or pca_state["max_tokens"] <= 0
        or pca_state["sampled_token_count"] <= 0
        or pca_state["sampled_token_count"] > pca_state["max_tokens"]
    ):
        raise ValueError("probe PCA metadata is incompatible")
    probe = SiglipPCAUpsampler(**config).to(device).eval()
    probe.load_state_dict(payload["state_dict"])
    probe.requires_grad_(False)
    return probe, payload
