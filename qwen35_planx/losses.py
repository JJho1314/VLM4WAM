"""Memory-bounded objectives for the grounded Qwen3.5 planner."""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def _finite(value: Tensor) -> Tensor:
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize(value: Tensor) -> Tensor:
    return F.normalize(_finite(value), dim=-1, eps=1e-12)


def _weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    finite_value = _finite(value)
    finite_weight = _finite(weight).clamp_min(0).to(finite_value)
    return (finite_value * finite_weight).sum() / finite_weight.sum().clamp_min(1e-12)


class _MemoryBoundedVisualCrossEntropy(torch.autograd.Function):
    """Recompute each visual-logit chunk instead of saving logits for backward."""

    @staticmethod
    def forward(
        ctx,
        hidden: Tensor,
        visual_weight: Tensor,
        targets: Tensor,
        chunk_size: int,
    ) -> tuple[Tensor, Tensor]:
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = targets.reshape(-1)
        loss_sum = hidden.new_zeros(())
        predictions = torch.empty_like(flat_targets)
        for start in range(0, flat_hidden.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_hidden.shape[0])
            logits = F.linear(flat_hidden[start:stop], visual_weight)
            loss_sum.add_(
                F.cross_entropy(
                    logits,
                    flat_targets[start:stop],
                    reduction="sum",
                )
            )
            predictions[start:stop] = logits.argmax(dim=-1)

        predictions = predictions.reshape(targets.shape)
        ctx.save_for_backward(hidden, visual_weight, targets)
        ctx.chunk_size = chunk_size
        ctx.mark_non_differentiable(predictions)
        return loss_sum / flat_hidden.shape[0], predictions

    @staticmethod
    def backward(
        ctx,
        grad_loss: Tensor | None,
        _grad_predictions: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, None, None]:
        hidden, visual_weight, targets = ctx.saved_tensors
        needs_hidden, needs_weight = ctx.needs_input_grad[:2]
        if grad_loss is None or not (needs_hidden or needs_weight):
            return None, None, None, None

        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = targets.reshape(-1)
        grad_hidden = torch.zeros_like(flat_hidden) if needs_hidden else None
        grad_weight = torch.zeros_like(visual_weight) if needs_weight else None
        scale = grad_loss / flat_hidden.shape[0]

        for start in range(0, flat_hidden.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, flat_hidden.shape[0])
            hidden_chunk = flat_hidden[start:stop]
            logits = F.linear(hidden_chunk, visual_weight)
            grad_logits = logits.softmax(dim=-1)
            target_column = flat_targets[start:stop].unsqueeze(-1)
            grad_logits.scatter_add_(
                dim=1,
                index=target_column,
                src=-torch.ones_like(target_column, dtype=grad_logits.dtype),
            )
            grad_logits.mul_(scale)
            if grad_hidden is not None:
                grad_hidden[start:stop] = grad_logits.matmul(visual_weight)
            if grad_weight is not None:
                grad_weight.addmm_(grad_logits.transpose(0, 1), hidden_chunk)

        hidden_gradient = (
            None if grad_hidden is None else grad_hidden.reshape(hidden.shape)
        )
        return hidden_gradient, grad_weight, None, None


def _chunked_visual_objective(
    hidden: Tensor,
    visual_weight: Tensor,
    targets: Tensor,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    if hidden.ndim < 2:
        raise ValueError("hidden must end in a feature dimension")
    if visual_weight.ndim != 2 or visual_weight.shape[1] != hidden.shape[-1]:
        raise ValueError(
            "visual_weight must have shape [visual_vocab, hidden_width]"
        )
    if targets.shape != hidden.shape[:-1]:
        raise ValueError("targets must match hidden's non-feature dimensions")
    if targets.dtype == torch.bool or targets.dtype.is_floating_point:
        raise TypeError("targets must contain integer visual-row indices")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1).to(device=hidden.device, dtype=torch.long)
    if flat_hidden.shape[0] == 0:
        raise ValueError("hidden must contain at least one prediction position")

    return _MemoryBoundedVisualCrossEntropy.apply(
        hidden,
        visual_weight,
        flat_targets.reshape(targets.shape),
        chunk_size,
    )


def chunked_visual_cross_entropy(
    hidden: Tensor,
    visual_weight: Tensor,
    targets: Tensor,
    *,
    chunk_size: int = 64,
) -> Tensor:
    """Cross-entropy over visual rows without retaining full-position logits.

    The largest temporary logits tensor is
    ``[min(chunk_size, remaining_positions), visual_vocab_size]``. No base
    language-vocabulary output head participates in this calculation.
    """

    loss, _ = _chunked_visual_objective(
        hidden,
        visual_weight,
        targets,
        chunk_size=chunk_size,
    )
    return loss


def dense_feature_loss(
    *,
    visual_regression: Tensor,
    target_code_embeddings: Tensor,
    semantic_features: Tensor,
    relevance_targets: Tensor,
    relevance_confidence: Tensor,
    predicted_phrase_embeddings: Tensor,
    phrase_embeddings: Tensor,
    field_mask: Tensor,
) -> Tensor:
    """Mean code, spatial-semantic, and phrase-anchor cosine regression."""

    if visual_regression.shape != target_code_embeddings.shape:
        raise ValueError("visual regression and codebook targets must align")
    if semantic_features.ndim != 4:
        raise ValueError("semantic_features must have shape [B,K,N,D]")
    batch, frames, tokens, text_dim = semantic_features.shape
    if relevance_targets.shape[:3] != (batch, frames, 3):
        raise ValueError("relevance_targets must have shape [B,K,3,N]")
    if relevance_targets.shape[-1] != tokens:
        raise ValueError("semantic and relevance token counts must match")
    if relevance_confidence.shape != (batch, frames, 3):
        raise ValueError("relevance_confidence must have shape [B,K,3]")
    if predicted_phrase_embeddings.shape != (batch, 3, text_dim):
        raise ValueError("predicted phrase embeddings must have shape [B,3,D]")
    if phrase_embeddings.shape != predicted_phrase_embeddings.shape:
        raise ValueError("cached and predicted phrase embeddings must align")
    if field_mask.shape != (batch, 3):
        raise ValueError("field_mask must have shape [B,3]")

    predicted_visual = _normalize(visual_regression)
    target_visual = _normalize(target_code_embeddings.to(visual_regression))
    visual_cosine = (
        1.0 - (predicted_visual * target_visual).sum(dim=-1)
    ).clamp_min(0)
    visual_term = visual_cosine.mean()

    maps = _finite(relevance_targets).clamp_min(0)
    confidence = _finite(relevance_confidence).clamp(0, 1)
    confident_maps = maps * confidence.unsqueeze(-1)
    cached_phrases = _normalize(phrase_embeddings.to(semantic_features))
    semantic_target = torch.einsum(
        "bkrt,brd->bktd",
        confident_maps,
        cached_phrases,
    )
    semantic_target = _normalize(semantic_target)
    predicted_semantic = _normalize(semantic_features)
    semantic_cosine = (
        1.0 - (predicted_semantic * semantic_target).sum(dim=-1)
    ).clamp_min(0)
    semantic_weight = confident_maps.sum(dim=2)
    semantic_term = _weighted_mean(semantic_cosine, semantic_weight)

    predicted_phrases = _normalize(predicted_phrase_embeddings)
    phrase_cosine = (
        1.0 - (predicted_phrases * cached_phrases).sum(dim=-1)
    ).clamp_min(0)
    phrase_term = _weighted_mean(phrase_cosine, field_mask.to(phrase_cosine))
    return (visual_term + semantic_term + phrase_term) / 3.0


def grounding_loss(
    relevance_logits: Tensor,
    relevance_targets: Tensor,
    relevance_confidence: Tensor,
    *,
    support_min: float = 0.01,
    support_max: float = 0.40,
    support_weight: float = 0.01,
) -> Tensor:
    """Confidence-weighted JS divergence plus effective-support hinge."""

    if relevance_logits.ndim != 4:
        raise ValueError("relevance_logits must have shape [B,K,3,N]")
    if relevance_targets.shape != relevance_logits.shape:
        raise ValueError("relevance targets and logits must align")
    if relevance_confidence.shape != relevance_logits.shape[:-1]:
        raise ValueError("relevance_confidence must have shape [B,K,3]")
    if not 0 <= support_min <= support_max <= 1:
        raise ValueError("support bounds must lie in [0,1]")
    if support_weight < 0:
        raise ValueError("support_weight must be non-negative")

    logits = _finite(relevance_logits)
    predicted = logits.softmax(dim=-1)
    target = _finite(relevance_targets).clamp_min(0).to(predicted)
    target_mass = target.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(target, 1.0 / target.shape[-1])
    target = torch.where(
        target_mass > 0,
        target / target_mass.clamp_min(1e-12),
        uniform,
    )
    midpoint = 0.5 * (predicted + target)
    tiny = torch.finfo(predicted.dtype).tiny
    predicted_log = predicted.clamp_min(tiny).log()
    target_log = target.clamp_min(tiny).log()
    midpoint_log = midpoint.clamp_min(tiny).log()
    js = 0.5 * (
        (predicted * (predicted_log - midpoint_log)).sum(dim=-1)
        + (target * (target_log - midpoint_log)).sum(dim=-1)
    )

    entropy = -(predicted * predicted_log).sum(dim=-1)
    support = entropy.exp() / predicted.shape[-1]
    support_hinge = (support_min - support).clamp_min(0)
    support_hinge = support_hinge + (support - support_max).clamp_min(0)
    confidence = _finite(relevance_confidence).clamp(0, 1).to(js)
    return _weighted_mean(js, confidence) + support_weight * _weighted_mean(
        support_hinge,
        confidence,
    )


def counterfactual_loss(
    semantic_features: Tensor,
    relevance: Tensor,
    positive_embeddings: Tensor,
    negative_embeddings: Tensor,
    counterfactual_mask: Tensor,
    *,
    fusion_gate: Tensor | None = None,
    margin: float = 0.2,
) -> Tensor:
    """Phrase-pooled cosine ranking against deterministic hard negatives."""

    if semantic_features.ndim != 4:
        raise ValueError("semantic_features must have shape [B,K,N,D]")
    batch, frames, tokens, text_dim = semantic_features.shape
    if relevance.shape != (batch, frames, 3, tokens):
        raise ValueError("relevance must have shape [B,K,3,N]")
    if positive_embeddings.shape != (batch, 3, text_dim):
        raise ValueError("positive_embeddings must have shape [B,3,D]")
    if negative_embeddings.ndim != 4 or negative_embeddings.shape[:2] != (
        batch,
        3,
    ):
        raise ValueError("negative_embeddings must have shape [B,3,M,D]")
    if negative_embeddings.shape[-1] != text_dim:
        raise ValueError("negative and semantic widths must match")
    if counterfactual_mask.shape != negative_embeddings.shape[:-1]:
        raise ValueError("counterfactual_mask must align with negatives")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    features = _normalize(semantic_features)
    if fusion_gate is not None:
        if fusion_gate.shape != (batch, frames, tokens, 1):
            raise ValueError("fusion_gate must have shape [B,K,N,1]")
        features = features * _finite(fusion_gate).clamp(0, 1).to(features)
    maps = _finite(relevance).clamp_min(0)
    maps = maps / maps.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    pooled = torch.einsum("bkrt,bktd->brd", maps, features)
    pooled = _normalize(pooled)
    positive = _normalize(positive_embeddings.to(pooled))
    negatives = _normalize(negative_embeddings.to(pooled))
    positive_score = (pooled * positive).sum(dim=-1)
    negative_score = torch.einsum("brd,brmd->brm", pooled, negatives)
    ranking = (
        margin - positive_score.unsqueeze(-1) + negative_score
    ).clamp_min(0)
    return _weighted_mean(ranking, counterfactual_mask.to(ranking))


def temporal_loss(relevance: Tensor, flow_targets: Tensor) -> Tensor:
    """Compare adjacent maps at valid cached DINO destination coordinates."""

    if relevance.ndim != 4:
        raise ValueError("relevance must have shape [B,K,3,N]")
    batch, frames, roles, tokens = relevance.shape
    if roles != 3:
        raise ValueError("relevance must contain source/target/action maps")
    if flow_targets.shape != (batch, frames - 1, tokens, 3):
        raise ValueError("flow_targets must have shape [B,K-1,N,3]")
    grid = math.isqrt(tokens)
    if grid * grid != tokens:
        raise ValueError("relevance token count must form a square grid")

    raw_flow = flow_targets.to(relevance)
    finite_flow = torch.isfinite(raw_flow).all(dim=-1)
    flow = _finite(raw_flow)
    source_index = torch.arange(tokens, device=relevance.device)
    source_y = torch.div(source_index, grid, rounding_mode="floor")
    source_x = source_index.remainder(grid)
    destination_x = source_x.view(1, 1, -1) + flow[..., 0].round().long()
    destination_y = source_y.view(1, 1, -1) + flow[..., 1].round().long()
    in_bounds = (
        (destination_x >= 0)
        & (destination_x < grid)
        & (destination_y >= 0)
        & (destination_y < grid)
    )
    destination_index = (
        destination_y.clamp(0, grid - 1) * grid
        + destination_x.clamp(0, grid - 1)
    )

    maps = _finite(relevance).clamp_min(0)
    source_maps = maps[:, :-1]
    destination_maps = maps[:, 1:]
    gather_index = destination_index.unsqueeze(2).expand(
        batch,
        frames - 1,
        roles,
        tokens,
    )
    warped_destination = torch.gather(
        destination_maps,
        dim=-1,
        index=gather_index,
    )
    difference = (source_maps - warped_destination).abs()
    confidence = flow[..., 2].clamp(0, 1)
    valid = finite_flow & in_bounds & (confidence > 0)
    weight = (confidence * valid).unsqueeze(2).expand_as(difference)
    return _weighted_mean(difference, weight)
