"""5-keyframe DINO-video plan head for the 4B lingbot-aligned planner (independent of the 2B stack).

Design decision (b): keep our 5-keyframe temporal plan by running ONE shared, warm-startable
``TaskTokenResampler`` (lingbot ``future_video_align_head``) once per keyframe, with a per-keyframe
positional embedding added to the query bank. Predicts ``[B, num_keyframes*num_backbone_tokens, dim_out]``
= ``[B, 5*256, 1024]`` = 1280 DINO-video patch tokens, matched to the teacher with plain MSE.

Per keyframe k:
    x        = cat( image_hidden(detached VLM image tokens),  latent_hidden_k(this kf's sem-plan latents) )
    queries  = query_embs[256] + keyframe_embed[k]
    preds_k  = TaskTokenResampler(x, queries)                       # [B, 256, 1024]

Warm-start (dims match at 4B: llm_hidden=2560, dim_out=1024, num_queries=256, 1 layer, heads4/dim_head32):
    resampler   <- ``future_video_align_head.projector.*``
    query_embs  <- ``future_video_align_embs``  [256, 2560]
    keyframe_embed: fresh (lingbot has a single future frame).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lingbot_resampler import TaskTokenResampler


class LingbotDinoPlanHead(nn.Module):
    def __init__(
        self,
        *,
        num_keyframes: int,
        num_latent_per_keyframe: int,
        num_backbone_tokens: int = 256,
        llm_hidden: int = 2560,
        dim_out: int = 1024,
        num_layers: int = 1,
        num_heads: int = 4,
        dim_head: int = 32,
        ff_mult: int = 1,
    ) -> None:
        super().__init__()
        self.num_keyframes = int(num_keyframes)
        self.num_latent_per_keyframe = int(num_latent_per_keyframe)
        self.num_backbone_tokens = int(num_backbone_tokens)
        self.llm_hidden = int(llm_hidden)
        self.dim_out = int(dim_out)

        self.resampler = TaskTokenResampler(
            dim_in=llm_hidden,
            dim_mid=llm_hidden,
            dim_head=dim_head,
            dim_out=dim_out,
            num_layers=num_layers,
            num_queries=num_backbone_tokens,
            num_heads=num_heads,
            ff_mult=ff_mult,
        )
        # learnable query bank [256, llm_hidden] (warm-startable from future_video_align_embs)
        self.query_embs = nn.Parameter(torch.randn(num_backbone_tokens, llm_hidden) / llm_hidden ** 0.5)
        # per-keyframe positional offset added to the query bank (fresh; lingbot has 1 future frame)
        self.keyframe_embed = nn.Parameter(torch.zeros(self.num_keyframes, llm_hidden))

    def forward(self, image_hidden: torch.Tensor, latent_hidden: torch.Tensor) -> torch.Tensor:
        # image_hidden: (B, N_img, H) — VLM image tokens (detach upstream);
        # latent_hidden: (B, num_keyframes*num_latent_per_keyframe, H) — this-episode sem-plan latents
        batch = image_hidden.shape[0]
        hidden = image_hidden.shape[-1]
        kf = self.num_keyframes
        expected = kf * self.num_latent_per_keyframe
        if latent_hidden.shape[1] != expected:
            raise RuntimeError(f"expected {expected} latent tokens, got {latent_hidden.shape[1]}")
        latents = latent_hidden.reshape(batch, kf, self.num_latent_per_keyframe, hidden)
        outs = []
        for k in range(kf):
            x = torch.cat([image_hidden, latents[:, k]], dim=1)  # (B, N_img+npl, H)
            q = (self.query_embs + self.keyframe_embed[k]).unsqueeze(0).expand(batch, -1, -1)  # (B,256,H)
            outs.append(self.resampler(x, q.to(x.dtype)))  # (B, 256, dim_out)
        return torch.stack(outs, dim=1).reshape(batch, kf * self.num_backbone_tokens, self.dim_out)

    @torch.no_grad()
    def load_lingbot_warmstart(self, state: dict) -> dict:
        """Warm-start from a lingbot-vla-v2 6b state_dict (or any dict containing the head keys).

        Returns a report ``{"resampler_missing":..., "resampler_unexpected":..., "query_loaded":bool}``.
        """
        marker = "future_video_align_head.projector."
        proj = {k.split(marker, 1)[1]: v for k, v in state.items() if marker in k}
        res = self.resampler.load_state_dict(proj, strict=False)
        query = [v for k, v in state.items() if k.endswith("future_video_align_embs")]
        query_loaded = False
        if query and query[0].shape == self.query_embs.shape:
            self.query_embs.data.copy_(query[0].to(self.query_embs.dtype))
            query_loaded = True
        return {
            "resampler_keys_loaded": len(proj),
            "resampler_missing": list(res.missing_keys),
            "resampler_unexpected": list(res.unexpected_keys),
            "query_loaded": query_loaded,
        }
