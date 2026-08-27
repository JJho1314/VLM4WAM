"""Faithful port of lingbot-vla-v2's align-head resampler (Apache-2.0).

Copied verbatim (only trimmed to the classes we need) from
``lingbot-vla-v2/lingbotvla/models/vla/vision_models/align_heads/resampler.py`` so that a trained
``future_video_align_head.projector.*`` state_dict loads into ``TaskTokenResampler`` 1:1 for warm-start.
Upstream itself is "modified from open_flamingo" (MIT). Keep the module/key names byte-identical.

Key layout (per layer N): ``layers.N.0.{norm1,norm2,to_q,to_kv,to_out}`` (PerceiverAttention, bias=False),
``layers.N.1.{0=LayerNorm,1=Linear,3=Linear}`` (FeedForward); plus ``proj_in1``(→queries),
``proj_in2``(→x), ``proj_out``, ``norm_out``. TaskTokenResampler has NO internal query param — queries
are supplied by the caller.
"""
import math

import torch
import torch.nn as nn


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """x: image/context features (b, n1, D); latents: query features (b, n2, D)."""
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out(out)


class TaskTokenResampler(nn.Module):
    """queries (external) cross-attend to cat(x, queries) through a Perceiver stack."""

    def __init__(
        self,
        dim_in=768,
        dim_mid=1024,
        dim_head=64,
        dim_out=1024,
        num_layers=8,
        num_queries=8,
        num_heads=16,
        ff_mult=4,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.proj_in1 = nn.Linear(dim_in, dim_mid)
        self.proj_in2 = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                        FeedForward(dim=dim_mid, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, queries):
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)

        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries

        queries = self.proj_out(queries)
        queries = self.norm_out(queries)
        return queries
