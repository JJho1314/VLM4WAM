import torch

from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_pca_probe import (
    fit_fixed_pca,
    project_fixed_pca,
    sample_pca_tokens,
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
