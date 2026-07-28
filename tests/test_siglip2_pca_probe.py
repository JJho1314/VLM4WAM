import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from qwen3_vl_semantic_planner.dinov3_da3_2b import (
    train_siglip2_pca_probe as train_probe,
)
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
from qwen3_vl_semantic_planner.dinov3_da3_2b.train_siglip2_pca_probe import (
    CachedFrameDataset,
    build_parser as build_probe_parser,
    split_episode_files,
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


def test_cached_frame_dataset_reads_nhwc_uint8(tmp_path: Path) -> None:
    path = tmp_path / "episode_000001.npy"
    np.save(path, np.full((2, 12, 10, 3), 128, dtype=np.uint8))
    dataset = CachedFrameDataset([path], virtual_length=2, seed=7)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        frame = dataset[0]

    assert frame.shape == (3, 12, 10)
    assert frame.dtype == torch.float32
    assert frame.mean().item() == pytest.approx(128 / 255)


def test_cached_frame_dataset_reads_nchw_uint8(tmp_path: Path) -> None:
    path = tmp_path / "episode_000002.npy"
    np.save(path, np.full((2, 3, 12, 10), 64, dtype=np.uint8))
    dataset = CachedFrameDataset([path], virtual_length=1, seed=11)

    frame = dataset[0]

    assert frame.shape == (3, 12, 10)
    assert frame.mean().item() == pytest.approx(64 / 255)


def test_cached_frame_dataset_rejects_ambiguous_channel_axes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "episode_000003.npy"
    np.save(path, np.zeros((2, 3, 12, 3), dtype=np.uint8))
    dataset = CachedFrameDataset([path], virtual_length=1, seed=13)

    with pytest.raises(ValueError, match="cannot infer cache layout"):
        dataset[0]


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


@pytest.mark.parametrize(
    ("model_name", "feature_dim", "error"),
    [
        ("another-siglip2-model", 1024, "model identity"),
        ("siglip2-large-patch16-256", 1152, "feature contract"),
    ],
)
def test_make_teachers_rejects_incompatible_model_contract(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    feature_dim: int,
    error: str,
) -> None:
    class FakeConfig:
        patch_size = 16

    class FakeModel:
        config = FakeConfig()

    class FakeTeacher:
        def __init__(
            self,
            *,
            input_size: int,
            grid_size: int,
            **_: object,
        ) -> None:
            self.input_size = input_size
            self.grid_size = grid_size
            self.feature_dim = feature_dim
            self.native_size = 256
            self.model = FakeModel()

    monkeypatch.setattr(
        train_probe,
        "Siglip2TargetEncoder",
        FakeTeacher,
    )

    with pytest.raises(ValueError, match=error):
        train_probe.make_teachers(
            Path("/models") / model_name,
            torch.device("cpu"),
        )


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
