from __future__ import annotations

import pytest
import torch


def test_compression_preserves_coverage_and_small_peak() -> None:
    from qwen35_planx.compression import compress_grounded_plan

    features = torch.arange(729.0).reshape(1, 1, 1, 729, 1)
    relevance = torch.zeros(1, 1, 1, 3, 729)
    relevance.select(-1, 17).fill_(1.0)

    plan = compress_grounded_plan(features, relevance, top_k=32)

    assert plan.tokens.shape == (1, 1, 1, 96, 1)
    assert plan.positions.shape == (1, 1, 1, 96, 2)
    assert plan.mask.shape == (1, 1, 1, 96)
    assert plan.relevance.shape == (1, 1, 1, 96)
    assert plan.source_indices.shape == (1, 1, 1, 96)
    assert plan.source_indices.select(-1, 64).item() == 17
    assert plan.tokens[0, 0, 0, 64, 0].item() == 17
    assert plan.mask[..., :65].all()
    assert not plan.mask[..., 65:].any()
    assert (plan.source_indices[..., :64] == -1).all()
    assert (plan.source_indices[..., 65:] == -1).all()
    assert torch.count_nonzero(plan.tokens[..., 65:, :]) == 0
    assert torch.count_nonzero(plan.positions[..., 65:, :]) == 0
    assert torch.count_nonzero(plan.relevance[..., 65:]) == 0


def test_fractional_area_overlap_and_relevance_weighting_are_exact() -> None:
    from qwen35_planx.compression import compress_grounded_plan

    boundary_feature = torch.zeros(1, 1, 1, 729, 1, dtype=torch.float64)
    boundary_feature[..., 3, 0] = 1.0
    no_relevance = torch.zeros(1, 1, 1, 3, 729, dtype=torch.float64)

    fractional = compress_grounded_plan(
        boundary_feature,
        no_relevance,
        top_k=0,
    )

    # The first 27/8-wide cell overlaps source column 3 by exactly 3/8.
    torch.testing.assert_close(
        fractional.tokens[0, 0, 0, 0, 0],
        torch.tensor(8.0 / 243.0, dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )

    weighted_feature = torch.zeros_like(boundary_feature)
    weighted_feature[..., 0, 0] = 1.0
    relevance = no_relevance.clone()
    relevance[..., 0, 0] = 1.0
    weighted = compress_grounded_plan(weighted_feature, relevance, top_k=0)

    # Cell area is 729/64. Doubling one unit-area patch yields 2/(729/64+1).
    expected = torch.tensor(128.0 / 793.0, dtype=torch.float64)
    torch.testing.assert_close(
        weighted.tokens[0, 0, 0, 0, 0],
        expected,
        rtol=0,
        atol=1e-15,
    )
    torch.testing.assert_close(
        weighted.relevance[0, 0, 0, 0],
        expected,
        rtol=0,
        atol=1e-15,
    )


def test_top_tokens_deduplicate_roles_break_ties_and_pad_fewer_peaks() -> None:
    from qwen35_planx.compression import compress_grounded_plan

    features = torch.arange(729.0).reshape(1, 1, 1, 729, 1)
    relevance = torch.zeros(1, 1, 1, 3, 729)
    relevance[..., 0, 10] = 0.7
    relevance[..., 1, 10] = 1.0
    relevance[..., 2, 10] = 0.9
    relevance[..., 0, 8] = 1.0
    relevance[..., 2, 2] = 1.0

    plan = compress_grounded_plan(features, relevance, top_k=5)

    assert plan.tokens.shape[-2:] == (69, 1)
    assert plan.source_indices[..., 64:].tolist() == [[[[2, 8, 10, -1, -1]]]]
    assert plan.tokens[..., 64:, 0].tolist() == [[[[2.0, 8.0, 10.0, 0.0, 0.0]]]]
    assert plan.relevance[..., 64:].tolist() == [[[[1.0, 1.0, 1.0, 0.0, 0.0]]]]
    assert plan.mask[..., 64:].tolist() == [[[[True, True, True, False, False]]]]


def test_positions_use_normalized_xy_patch_centers_and_stay_in_bounds() -> None:
    from qwen35_planx.compression import compress_grounded_plan

    features = torch.zeros(1, 1, 1, 729, 1, dtype=torch.float64)
    relevance = torch.zeros(1, 1, 1, 3, 729, dtype=torch.float64)
    relevance[..., 0, 17] = 1.0
    plan = compress_grounded_plan(features, relevance, top_k=1)

    # Coverage index 1 is target row 0, column 1 under raster order. Fractional
    # boundary weights pool the original discrete patch centers.
    torch.testing.assert_close(
        plan.positions[0, 0, 0, 1],
        torch.tensor([91.0 / 486.0, 31.0 / 486.0], dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )
    torch.testing.assert_close(
        plan.positions[0, 0, 0, 64],
        torch.tensor([17.5 / 27.0, 0.5 / 27.0], dtype=torch.float64),
        rtol=0,
        atol=1e-15,
    )
    assert torch.all((plan.positions >= 0) & (plan.positions <= 1))


def test_gradients_reach_coverage_selected_features_and_relevance() -> None:
    from qwen35_planx.compression import compress_grounded_plan

    torch.manual_seed(19)
    features = torch.randn(
        1,
        1,
        1,
        729,
        2,
        dtype=torch.float64,
        requires_grad=True,
    )
    relevance = torch.zeros(1, 1, 1, 3, 729, dtype=torch.float64)
    relevance[..., 0, 17] = 1.0
    relevance[..., 1, 211] = 0.5
    relevance.requires_grad_()

    plan = compress_grounded_plan(features, relevance, top_k=2)
    loss = plan.tokens.sum() + plan.positions.sum() + plan.relevance.sum()
    loss.backward()

    assert features.grad is not None
    assert relevance.grad is not None
    assert torch.all(features.grad != 0)
    assert features.grad[..., 17, :].min() > features.grad[..., 18, :].max()
    assert torch.count_nonzero(relevance.grad) > 0


@pytest.mark.parametrize(
    ("features", "relevance", "error", "message"),
    (
        (
            torch.zeros(1, 1, 729, 2),
            torch.zeros(1, 1, 1, 3, 729),
            ValueError,
            "features",
        ),
        (
            torch.zeros(1, 1, 1, 728, 2),
            torch.zeros(1, 1, 1, 3, 729),
            ValueError,
            "729",
        ),
        (
            torch.zeros(1, 1, 1, 729, 2),
            torch.zeros(1, 2, 1, 3, 729),
            ValueError,
            "leading",
        ),
        (
            torch.zeros(1, 1, 1, 729, 2),
            torch.zeros(1, 1, 1, 2, 729),
            ValueError,
            "relevance",
        ),
        (
            torch.zeros(1, 1, 1, 729, 0),
            torch.zeros(1, 1, 1, 3, 729),
            ValueError,
            "width",
        ),
        (
            torch.zeros(0, 1, 1, 729, 2),
            torch.zeros(0, 1, 1, 3, 729),
            ValueError,
            "leading",
        ),
        (
            torch.zeros(1, 1, 1, 729, 2, dtype=torch.long),
            torch.zeros(1, 1, 1, 3, 729),
            TypeError,
            "floating",
        ),
        (
            torch.zeros(1, 1, 1, 729, 2),
            torch.zeros(1, 1, 1, 3, 729, dtype=torch.float64),
            TypeError,
            "dtype",
        ),
    ),
)
def test_compression_rejects_invalid_shapes_and_dtypes(
    features: torch.Tensor,
    relevance: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    from qwen35_planx.compression import compress_grounded_plan

    with pytest.raises(error, match=message):
        compress_grounded_plan(features, relevance)


def test_compression_requires_tensor_inputs() -> None:
    from qwen35_planx.compression import compress_grounded_plan

    features = torch.zeros(1, 1, 1, 729, 2)
    relevance = torch.zeros(1, 1, 1, 3, 729)

    with pytest.raises(TypeError, match="features"):
        compress_grounded_plan(features.tolist(), relevance)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="relevance"):
        compress_grounded_plan(features, relevance.tolist())  # type: ignore[arg-type]


@pytest.mark.parametrize(("tensor_name", "value"), (("features", torch.nan), ("relevance", torch.inf)))
def test_compression_rejects_nonfinite_inputs(
    tensor_name: str,
    value: float,
) -> None:
    from qwen35_planx.compression import compress_grounded_plan

    features = torch.zeros(1, 1, 1, 729, 2)
    relevance = torch.zeros(1, 1, 1, 3, 729)
    if tensor_name == "features":
        features[..., 0, 0] = value
    else:
        relevance[..., 0, 0] = value

    with pytest.raises(ValueError, match="finite"):
        compress_grounded_plan(features, relevance)


def test_compression_rejects_negative_relevance() -> None:
    from qwen35_planx.compression import compress_grounded_plan

    features = torch.zeros(1, 1, 1, 729, 2)
    relevance = torch.zeros(1, 1, 1, 3, 729)
    relevance[..., 0, 0] = -0.1

    with pytest.raises(ValueError, match="non-negative"):
        compress_grounded_plan(features, relevance)


def test_compression_rejects_device_mismatch() -> None:
    from qwen35_planx.compression import compress_grounded_plan

    features = torch.zeros(1, 1, 1, 729, 2)
    relevance = torch.zeros(1, 1, 1, 3, 729, device="meta")

    with pytest.raises(ValueError, match="device"):
        compress_grounded_plan(features, relevance)


@pytest.mark.parametrize("top_k", (-1, 33, 1.5, True))
def test_compression_rejects_invalid_top_k(top_k: object) -> None:
    from qwen35_planx.compression import compress_grounded_plan

    features = torch.zeros(1, 1, 1, 729, 2)
    relevance = torch.zeros(1, 1, 1, 3, 729)

    with pytest.raises((TypeError, ValueError), match="top_k"):
        compress_grounded_plan(features, relevance, top_k=top_k)  # type: ignore[arg-type]
