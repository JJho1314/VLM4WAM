from __future__ import annotations

import pytest
import torch

from qwen35_baton.losses import compute_spatial_wsa_loss
from qwen35_baton.model import SPATIAL_DIM, SPATIAL_LAYERS, BatonQwen35Planner
from test_qwen35_baton_model import ADDED_TOKEN_IDS, TinyQwen, make_batch


def make_variant(
    *, residual: bool, spatial: bool, current_context: bool = False
) -> BatonQwen35Planner:
    torch.manual_seed(0)
    return BatonQwen35Planner(
        TinyQwen(width=16),
        added_token_ids=ADDED_TOKEN_IDS,
        current_context=current_context,
        residual=residual,
        spatial=spatial,
    )


def test_default_planner_has_no_variant_modules() -> None:
    planner = make_variant(residual=False, spatial=False)

    names = {name for name, _ in planner.named_parameters()}

    assert not any(
        name.startswith(("current_projection.", "spa_query_tower.", "spa_mlp."))
        for name in names
    )
    assert planner(make_batch()).spatial is None


def test_residual_planner_starts_at_copy_current_baseline() -> None:
    planner = make_variant(residual=True, spatial=False)
    current = torch.randn(2, 2, 256, 1024)

    output = planner(make_batch(), current_features=current)

    expected = current.unsqueeze(2).expand(-1, -1, 4, -1, -1)
    assert torch.allclose(output.positive, expected)


def test_residual_planner_uses_current_features_after_training_starts() -> None:
    planner = make_variant(residual=True, spatial=False)
    torch.nn.init.normal_(planner.sem_mlp[-1].weight, std=0.1)
    batch = make_batch()
    current = torch.randn(2, 2, 256, 1024)
    shifted = current.clone()
    shifted[:, :, :, :8] += 5.0

    base = planner(batch, current_features=current).positive
    moved = planner(batch, current_features=shifted).positive

    # The shift reaches the output through both the skip and the tower context.
    delta = (moved - base) - (shifted - current).unsqueeze(2)
    assert delta.abs().max() > 0


def test_residual_planner_requires_current_features() -> None:
    planner = make_variant(residual=True, spatial=False)

    with pytest.raises(ValueError, match="current_features"):
        planner(make_batch())


def test_non_residual_planner_rejects_current_features() -> None:
    planner = make_variant(residual=False, spatial=False)

    with pytest.raises(ValueError, match="current_features"):
        planner(make_batch(), current_features=torch.zeros(2, 2, 256, 1024))


def test_context_planner_reads_current_features_without_skip() -> None:
    planner = make_variant(residual=False, spatial=False, current_context=True)
    batch = make_batch()
    current = torch.randn(2, 2, 256, 1024)

    base = planner(batch, current_features=current).positive
    moved = planner(batch, current_features=current + 1.0).positive

    assert not torch.allclose(base, current.unsqueeze(2).expand_as(base))
    assert not torch.allclose(base, moved)


def test_spatial_head_predicts_wsa_layers_per_camera_and_keyframe() -> None:
    planner = make_variant(residual=False, spatial=True)

    output = planner(make_batch())

    assert output.spatial is not None
    assert output.spatial.shape == (2, 2, 4, 256, SPATIAL_LAYERS, SPATIAL_DIM)


def test_spatial_wsa_loss_is_zero_for_exact_prediction() -> None:
    target = torch.randn(3, 5, 4, 32)

    loss = compute_spatial_wsa_loss(target.clone(), target, torch.ones(4))

    assert loss.total.item() == pytest.approx(0.0, abs=1e-5)


def test_spatial_wsa_loss_weights_layers() -> None:
    target = torch.randn(3, 5, 2, 32)
    prediction = target.clone()
    prediction[..., 1, :] = -target[..., 1, :]

    light = compute_spatial_wsa_loss(prediction, target, torch.tensor([1.0, 1.0]))
    heavy = compute_spatial_wsa_loss(prediction, target, torch.tensor([1.0, 2.0]))

    assert heavy.total > light.total > 0
    assert light.cosine.item() == pytest.approx(1.0, abs=1e-4)
