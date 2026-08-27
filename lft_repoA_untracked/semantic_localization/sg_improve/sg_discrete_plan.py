"""Option 6: light discretization of the SigLIP semantic plan (VQ) + PARALLEL code prediction by the
Qwen3-VL-2B planner -- gets Plan-X's "discrete text-aligned semantic tokens" benefit WITHOUT training
an autoregressive generator.

Idea:
  - Build a VQ codebook over GT SigLIP2 features. Because SigLIP is text-aligned, the codes inherit
    that alignment -> a discrete, text-grounded semantic vocabulary (a cheap TA-Tok surrogate).
  - The Qwen3-VL-2B planner keeps its per-keyframe-token query features, but instead of regressing a
    continuous 1024-d vector it predicts, IN PARALLEL (one classification per token, no autoregression),
    a code index over the codebook. Loss = cross-entropy vs the GT feature's code (+ optional recon).
  - At inference: argmax -> codebook vector -> a 1024-d plan fed to the WAM UNCHANGED (semantic_adapter
    in_dim stays 1024). So the WAM side needs no change; only the plan becomes discrete/compositional.

Drop-in for the qwen3_vl_semantic_planner: replace the continuous plan regression head with
`ParallelCodePlanHead` and the plan MSE with `discrete_plan_loss`. See README.md.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPVQ(nn.Module):
    """EMA vector-quantizer over SigLIP features. Codebook [K, D]. quantize() gives target codes for the
    planner to classify; embed() maps predicted codes back to 1024-d plan vectors for the WAM."""

    def __init__(self, num_codes: int = 2048, dim: int = 1024, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_codes, self.dim, self.decay, self.eps = num_codes, dim, decay, eps
        self.register_buffer("codebook", F.normalize(torch.randn(num_codes, dim), dim=-1))
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", self.codebook.clone())

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., D] (L2-normalized recommended) -> code indices [...]. Nearest codebook entry."""
        flat = F.normalize(x.reshape(-1, self.dim).float(), dim=-1)
        d = flat @ self.codebook.t()            # cosine similarity (codebook is unit-norm)
        return d.argmax(dim=-1).reshape(x.shape[:-1])

    def embed(self, codes: torch.Tensor) -> torch.Tensor:
        """code indices [...] -> 1024-d plan vectors [..., D] for the WAM."""
        return F.embedding(codes, self.codebook)

    @torch.no_grad()
    def ema_update(self, x: torch.Tensor, codes: torch.Tensor) -> None:
        """Optional online codebook refresh from a batch of GT features x [...,D] & their codes [...]."""
        flat = F.normalize(x.reshape(-1, self.dim).float(), dim=-1)
        idx = codes.reshape(-1)
        onehot = F.one_hot(idx, self.num_codes).type(flat.dtype)         # [N, K]
        n = onehot.sum(0)
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        dw = onehot.t() @ flat                                           # [K, D]
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        upd = self.ema_w / (self.cluster_size[:, None] + self.eps)
        self.codebook.copy_(F.normalize(upd, dim=-1))


class ParallelCodePlanHead(nn.Module):
    """Planner query features -> per-token code logits (parallel classification, NOT autoregressive)."""

    def __init__(self, query_dim: int, num_codes: int = 2048, hid: int = 1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(query_dim, hid), nn.GELU(), nn.Linear(hid, num_codes))

    def forward(self, query_feat: torch.Tensor) -> torch.Tensor:
        return self.net(query_feat)             # [..., num_codes]


def discrete_plan_loss(
    logits: torch.Tensor,        # [B, V, K, P, num_codes]  planner code logits (parallel)
    gt_siglip: torch.Tensor,     # [B, V, K, P, D]          GT SigLIP2 features (the plan target)
    vq: SigLIPVQ,
    *,
    ema: bool = True,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CE(predicted code | token) vs the GT feature's nearest code. Returns (loss, target_codes)."""
    with torch.no_grad():
        codes = vq.quantize(gt_siglip)          # [B, V, K, P]
        if ema:
            vq.ema_update(gt_siglip, codes)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), codes.reshape(-1), label_smoothing=label_smoothing
    )
    return loss, codes


@torch.no_grad()
def predict_plan_vectors(logits: torch.Tensor, vq: SigLIPVQ) -> torch.Tensor:
    """Inference: argmax code logits -> 1024-d codebook plan vectors for the WAM. [.., num_codes] -> [.., D]."""
    return vq.embed(logits.argmax(dim=-1))
