from __future__ import annotations

import torch
import torch.nn.functional as F


def test_chunked_visual_ce_matches_dense_reference_and_gradients() -> None:
    from qwen35_planx.losses import chunked_visual_cross_entropy

    torch.manual_seed(7)
    hidden = torch.randn(2, 17, 8, requires_grad=True)
    weight = torch.randn(31, 8, requires_grad=True)
    targets = torch.randint(0, 31, (2, 17))

    actual = chunked_visual_cross_entropy(
        hidden,
        weight,
        targets,
        chunk_size=5,
    )
    expected = F.cross_entropy(
        hidden.detach().reshape(-1, 8) @ weight.detach().T,
        targets.reshape(-1),
    )
    torch.testing.assert_close(actual, expected)

    actual.backward()
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_weight_grad = weight.grad.detach().clone()
    hidden.grad = None
    weight.grad = None
    dense = F.cross_entropy(
        hidden.reshape(-1, 8) @ weight.T,
        targets.reshape(-1),
    )
    dense.backward()
    torch.testing.assert_close(actual_hidden_grad, hidden.grad)
    torch.testing.assert_close(actual_weight_grad, weight.grad)


def test_chunked_visual_ce_bf16_matches_fp32_reference_and_gradients() -> None:
    from qwen35_planx.losses import chunked_visual_cross_entropy

    torch.manual_seed(91)
    hidden_fp32 = torch.randn(2, 33, 256, requires_grad=True)
    weight_fp32 = torch.randn(4096, 256, requires_grad=True)
    targets = torch.randint(0, 4096, (2, 33))
    reference = F.cross_entropy(
        hidden_fp32.reshape(-1, 256) @ weight_fp32.T,
        targets.reshape(-1),
    )
    reference.backward()
    expected_hidden_grad = hidden_fp32.grad.detach().clone()
    expected_weight_grad = weight_fp32.grad.detach().clone()

    hidden = hidden_fp32.detach().to(torch.bfloat16).requires_grad_(True)
    weight = weight_fp32.detach().to(torch.bfloat16).requires_grad_(True)
    actual = chunked_visual_cross_entropy(hidden, weight, targets, chunk_size=7)
    actual.backward()

    torch.testing.assert_close(actual.float(), reference, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(
        hidden.grad.float(), expected_hidden_grad, atol=3e-3, rtol=3e-2
    )
    torch.testing.assert_close(
        weight.grad.float(), expected_weight_grad, atol=3e-3, rtol=3e-2
    )


def test_chunked_visual_ce_never_materializes_more_than_requested_positions(
    monkeypatch,
) -> None:
    import qwen35_planx.losses as losses

    hidden = torch.randn(3, 11, 7)
    weight = torch.randn(29, 7)
    targets = torch.randint(0, 29, (3, 11))
    seen: list[int] = []
    original = losses.F.linear

    def recording_linear(input: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        if rows is weight:
            seen.append(int(input.shape[0]))
        return original(input, rows)

    monkeypatch.setattr(losses.F, "linear", recording_linear)
    losses.chunked_visual_cross_entropy(hidden, weight, targets, chunk_size=4)

    assert seen
    assert max(seen) <= 4
    assert sum(seen) == hidden.shape[0] * hidden.shape[1]


def test_chunked_visual_ce_retains_no_vocabulary_sized_activations() -> None:
    from qwen35_planx.losses import chunked_visual_cross_entropy

    vocabulary_size = 65_536
    hidden = torch.randn(2, 7, 4, requires_grad=True)
    weight = torch.randn(vocabulary_size, 4, requires_grad=True)
    targets = torch.randint(0, vocabulary_size, (2, 7))
    input_storages = {
        hidden.untyped_storage().data_ptr(),
        weight.untyped_storage().data_ptr(),
        targets.untyped_storage().data_ptr(),
    }
    saved: list[torch.Tensor] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = chunked_visual_cross_entropy(
            hidden,
            weight,
            targets,
            chunk_size=5,
        )
    retained_activations = [
        tensor
        for tensor in saved
        if vocabulary_size in tensor.shape
        and tensor.untyped_storage().data_ptr() not in input_storages
    ]

    assert retained_activations == []
    loss.backward()
    assert hidden.grad is not None
    assert weight.grad is not None


def test_dense_feature_loss_is_zero_for_exact_normalized_targets() -> None:
    from qwen35_planx.losses import dense_feature_loss

    code_targets = F.normalize(torch.randn(1, 8, 5), dim=-1)
    phrases = F.normalize(torch.randn(1, 3, 4), dim=-1)
    relevance = torch.zeros(1, 2, 3, 4)
    relevance[:, :, 0, 0] = 1
    relevance[:, :, 1, 1] = 1
    relevance[:, :, 2, 2] = 1
    confidence = torch.ones(1, 2, 3)
    semantic_target = torch.einsum("bkrt,brd->bktd", relevance, phrases)
    semantic_target = F.normalize(semantic_target, dim=-1)

    actual = dense_feature_loss(
        visual_regression=code_targets,
        target_code_embeddings=code_targets,
        semantic_features=semantic_target,
        relevance_targets=relevance,
        relevance_confidence=confidence,
        predicted_phrase_embeddings=phrases,
        phrase_embeddings=phrases,
        field_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    torch.testing.assert_close(actual, torch.tensor(0.0), atol=1e-6, rtol=0)


def test_zero_confidence_and_masks_are_finite_and_contribute_zero() -> None:
    from qwen35_planx.losses import dense_feature_loss, grounding_loss

    prediction = F.normalize(torch.randn(1, 4, 3), dim=-1)
    semantic = F.normalize(torch.randn(1, 1, 4, 2), dim=-1)
    phrases = F.normalize(torch.randn(1, 3, 2), dim=-1)
    relevance = torch.full((1, 1, 3, 4), float("nan"))
    confidence = torch.zeros(1, 1, 3)
    field_mask = torch.zeros(1, 3, dtype=torch.bool)

    dense = dense_feature_loss(
        visual_regression=prediction,
        target_code_embeddings=prediction,
        semantic_features=semantic,
        relevance_targets=relevance,
        relevance_confidence=confidence,
        predicted_phrase_embeddings=phrases,
        phrase_embeddings=torch.full_like(phrases, float("nan")),
        field_mask=field_mask,
    )
    logits = torch.randn(1, 1, 3, 4, requires_grad=True)
    grounding = grounding_loss(logits, relevance, confidence)
    (dense + grounding).backward()

    assert torch.isfinite(dense)
    assert torch.isfinite(grounding)
    torch.testing.assert_close(dense, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(grounding, torch.tensor(0.0), atol=1e-6, rtol=0)
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 0


def test_grounding_loss_matches_a_supported_teacher_distribution() -> None:
    from qwen35_planx.losses import grounding_loss

    target = torch.full((1, 1, 3, 16), 1e-6)
    target[..., :2] = 1
    target = target / target.sum(dim=-1, keepdim=True)
    logits = target.log()
    confidence = torch.ones(1, 1, 3)

    actual = grounding_loss(logits, target, confidence)

    torch.testing.assert_close(actual, torch.tensor(0.0), atol=1e-6, rtol=0)


def test_counterfactual_margin_uses_only_valid_same_suite_negatives() -> None:
    from qwen35_planx.losses import counterfactual_loss

    features = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]]]],
    )
    relevance = torch.zeros(1, 1, 3, 2)
    relevance[..., 0] = 1
    positives = torch.tensor([[[1.0, 0.0]]]).expand(1, 3, 2)
    negatives = torch.tensor([[[[0.0, 1.0]]]]).expand(1, 3, 1, 2)
    valid = torch.ones(1, 3, 1, dtype=torch.bool)

    separated = counterfactual_loss(
        features,
        relevance,
        positives,
        negatives,
        valid,
        margin=0.2,
    )
    wrong = counterfactual_loss(
        features,
        relevance,
        torch.tensor([[[0.0, 1.0]]]).expand(1, 3, 2),
        torch.tensor([[[[1.0, 0.0]]]]).expand(1, 3, 1, 2),
        valid,
        margin=0.2,
    )
    masked = counterfactual_loss(
        features,
        relevance,
        positives,
        torch.full_like(negatives, float("nan")),
        torch.zeros_like(valid),
        margin=0.2,
    )

    torch.testing.assert_close(separated, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(wrong, torch.tensor(1.2), atol=1e-6, rtol=0)
    torch.testing.assert_close(masked, torch.tensor(0.0), atol=1e-6, rtol=0)
    assert torch.isfinite(masked)


def test_temporal_loss_follows_flow_and_ignores_invalid_destinations() -> None:
    from qwen35_planx.losses import temporal_loss

    maps = torch.zeros(1, 2, 3, 9)
    maps[:, 0, :, 4] = 1
    maps[:, 1, :, 5] = 1
    flow = torch.zeros(1, 1, 9, 3)
    flow[..., 0] = 1
    flow[..., 2] = 1

    consistent = temporal_loss(maps, flow)
    invalid_flow = flow.clone()
    invalid_flow[..., 0] = 100
    ignored = temporal_loss(maps, invalid_flow)

    torch.testing.assert_close(consistent, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(ignored, torch.tensor(0.0), atol=1e-6, rtol=0)
    assert torch.isfinite(ignored)
