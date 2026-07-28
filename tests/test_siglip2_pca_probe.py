import inspect
from pathlib import Path

import pytest
import torch

from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_pca_probe import (
    SiglipPCAUpsampler,
    fit_fixed_pca,
    load_validated_probe,
    multiscale_gradient_loss,
    pca_target_images,
    project_fixed_pca,
    sample_pca_tokens,
    validation_gate_passed,
    validation_metrics,
)


def _valid_probe_payload() -> dict[str, object]:
    return {
        "accepted": True,
        "model_name": "siglip2-large-patch16-256",
        "feature_layer": "penultimate_spatial",
        "low_input_size": 256,
        "high_input_size": 512,
        "high_grid_size": 32,
        "state_dict": {},
        "config": {
            "in_dim": 1024,
            "hidden_dim": 32,
            "grid_size": 16,
            "output_size": 256,
        },
        "pca_state": {
            "mean": torch.zeros(1024),
            "components": torch.eye(1024)[:3],
            "display_low": torch.zeros(3),
            "display_high": torch.ones(3),
            "feature_dim": 1024,
            "component_sign_rule": "largest_absolute_loading_positive",
            "max_tokens": 64,
            "seed": 5,
            "sampled_token_count": 64,
        },
        "validation_metrics": {
            "probe_l1": 0.1,
            "baseline_l1": 0.2,
            "probe_gradient": 0.1,
            "baseline_gradient": 0.2,
        },
    }


def test_sample_pca_tokens_is_bounded_and_deterministic() -> None:
    features = torch.arange(8 * 5, dtype=torch.float32).reshape(2, 4, 5)

    first = sample_pca_tokens(features, max_tokens=5, seed=17)
    second = sample_pca_tokens(features, max_tokens=5, seed=17)

    assert first.shape == (5, 5)
    torch.testing.assert_close(first, second)


def test_fit_fixed_pca_anchors_component_signs() -> None:
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(4, 16, 8, generator=generator)

    state = fit_fixed_pca(features, max_tokens=64, seed=9)
    components = state["components"]
    anchor_columns = components.abs().argmax(dim=1)

    assert state["mean"].shape == (8,)
    assert components.shape == (3, 8)
    assert state["component_sign_rule"] == "largest_absolute_loading_positive"
    assert torch.all(components[torch.arange(3), anchor_columns] > 0)


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
    baseline[..., ::2] = 0
    probe = torch.full_like(target, 0.25)
    metrics = validation_metrics(probe, baseline, target)

    assert metrics["probe_l1"] < metrics["baseline_l1"]
    assert metrics["probe_gradient"] < metrics["baseline_gradient"]
    assert validation_gate_passed(metrics)
    assert not validation_gate_passed(
        {
            **metrics,
            "probe_gradient": metrics["baseline_gradient"] + 1.0,
        }
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


def test_load_validated_probe_rejects_accepted_checkpoint_with_failed_metrics(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "failed_metrics.pt"
    payload = _valid_probe_payload()
    payload["validation_metrics"] = {
        "probe_l1": 0.3,
        "baseline_l1": 0.2,
        "probe_gradient": 0.1,
        "baseline_gradient": 0.2,
    }
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="validation gate"):
        load_validated_probe(
            checkpoint,
            expected_model_name="siglip2-large-patch16-256",
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("field", ["seed", "sampled_token_count"])
def test_load_validated_probe_requires_complete_pca_metadata(
    tmp_path: Path,
    field: str,
) -> None:
    checkpoint = tmp_path / f"missing_{field}.pt"
    payload = _valid_probe_payload()
    del payload["pca_state"][field]  # type: ignore[index]
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="PCA state is missing"):
        load_validated_probe(
            checkpoint,
            expected_model_name="siglip2-large-patch16-256",
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", "not-an-integer"),
        ("max_tokens", 0),
        ("sampled_token_count", 0),
        ("sampled_token_count", 65),
    ],
)
def test_load_validated_probe_rejects_invalid_pca_metadata(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    checkpoint = tmp_path / f"invalid_{field}.pt"
    payload = _valid_probe_payload()
    payload["pca_state"][field] = value  # type: ignore[index]
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="PCA metadata"):
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
