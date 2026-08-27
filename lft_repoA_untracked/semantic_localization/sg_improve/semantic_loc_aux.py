"""Auxiliary spatial-grounding loss for SG-WAM (Option 2, no autoregressive needed).

Root cause found in this repo's probes: the planner-predicted SigLIP `semantic_plan` encodes semantic
CONTENT but is NOT target-localizable (linear probe: SG≈baseline, p>0.1). SigLIP penultimate DOES carry
object position (a trained FiLM+conv head recovers it, distilling CLIPSeg), but nothing FORCES the plan
to expose it. This module adds that force as an auxiliary loss on the *planner's predicted plan*:
predict the target-object mask from the plan tokens + noun text, distilling CLIPSeg pseudo-GT.
This is MaskWAM's "predict the target's spatial extent" idea, ported to continuous SigLIP conditioning
via an auxiliary loss (no mask input, no discrete/autoregressive tokens).

Integrate at 3 points (see INTEGRATION.md):
  1. JointVLMGEActModel.__init__:  self.loc_head = SemanticLocalizationHead(plan_dim=1024)
  2. JointVLMGEActModel.forward (after `semantic_plan, depth_plan, planner_losses = planner_result`):
        loc = semantic_localization_loss(self.loc_head, semantic_plan, batch["target_noun_emb"],
                                         batch["target_masks"], num_keyframes=self.num_keyframes,
                                         tokens_per_keyframe=self.tokens_per_keyframe)
        planner_losses = {**planner_losses, "loc_loss": loc}
  3. combine_joint_training_loss: add  + float(loc_loss_weight) * planner_losses["loc_loss"]
Supervision (target_noun_emb, target_masks) is precomputed offline by precompute_target_masks.py.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLocalizationHead(nn.Module):
    """FiLM(text) + conv decoder on SigLIP plan tokens -> per-patch target-mask logit.
    Same design as the standalone loc-head that empirically recovers position from SigLIP penultimate
    (~1.7M params, frozen backbone). Input plan tokens are [N, P, D] (P=16*16 patches per keyframe)."""

    def __init__(self, plan_dim: int = 1024, hid: int = 256, text_dim: int = 512):
        super().__init__()
        self.inp = nn.Conv2d(plan_dim, hid, 1)
        self.film = nn.Linear(text_dim, 2 * hid)
        self.net = nn.Sequential(
            nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.GELU(),
        )
        self.out = nn.Conv2d(hid, 1, 1)

    def forward(self, plan_tokens: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # plan_tokens [N, P, D] with P a perfect square (e.g. 256 -> 16x16); text_emb [N, text_dim]
        n, p, d = plan_tokens.shape
        g = int(round(math.sqrt(p)))
        if g * g != p:
            raise ValueError(f"plan tokens per keyframe must be square, got {p}")
        x = plan_tokens.transpose(1, 2).reshape(n, d, g, g)
        x = self.inp(x)
        gamma, beta = self.film(text_emb).chunk(2, dim=-1)
        x = x * (1 + gamma[..., None, None]) + beta[..., None, None]
        return self.out(self.net(x)).squeeze(1)  # [N, g, g]


def semantic_localization_loss(
    head: SemanticLocalizationHead,
    semantic_plan: torch.Tensor,     # [B, V, K, P, D] planner-predicted SigLIP plan (keep graph!)
    target_noun_emb: torch.Tensor,   # [B, text_dim]   CLIP text emb of the instruction's target noun
    target_masks: torch.Tensor,      # [B, V, K, g, g]  CLIPSeg soft pseudo-GT of the target on each keyframe
    *,
    num_keyframes: int,
    tokens_per_keyframe: int,
    mask_valid: torch.Tensor | None = None,   # [B] optional: 0 for language-clear samples to skip
) -> torch.Tensor:
    """BCE(target-mask | plan tokens, noun). Gradients flow into `semantic_plan` -> planner/Qwen,
    forcing the predicted SigLIP features to become target-localizable. Returns a scalar loss."""
    b, v, k, p, d = semantic_plan.shape
    if k != num_keyframes or p != tokens_per_keyframe:
        raise ValueError(f"semantic_plan [B,V,K,P,D] mismatch: got K={k},P={p}")
    tokens = semantic_plan.reshape(b * v * k, p, d)
    text = target_noun_emb[:, None, None, :].expand(b, v, k, -1).reshape(b * v * k, -1)
    logits = head(tokens, text)                       # [B*V*K, g, g]
    gt = target_masks.reshape(b * v * k, *target_masks.shape[-2:]).to(logits.dtype)
    per = F.binary_cross_entropy_with_logits(logits, gt, reduction="none").mean(dim=(-1, -2))  # [B*V*K]
    if mask_valid is not None:
        w = mask_valid[:, None, None].expand(b, v, k).reshape(-1).to(per.dtype)
        return (per * w).sum() / (w.sum() + 1e-6)
    return per.mean()
