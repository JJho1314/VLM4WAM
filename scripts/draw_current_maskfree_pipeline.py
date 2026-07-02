#!/usr/bin/env python3
"""Draw a clean architecture diagram for the current mask-free latent grounding method."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parents[1]
PNG = OUT_DIR / "current_maskfree_latent_grounding_pipeline_clean_v7.png"
PDF = OUT_DIR / "current_maskfree_latent_grounding_pipeline_clean_v7.pdf"


COLORS = {
    "green": "#D9F7E8",
    "green_edge": "#059669",
    "blue": "#DBEAFE",
    "blue_edge": "#2563EB",
    "purple": "#EDE9FE",
    "purple_edge": "#7C3AED",
    "amber": "#FEF3C7",
    "amber_edge": "#D97706",
    "orange": "#FFEDD5",
    "orange_edge": "#EA580C",
    "gray": "#F8FAFC",
    "gray_edge": "#94A3B8",
    "red": "#FEE2E2",
    "red_edge": "#DC2626",
    "ink": "#111827",
    "muted": "#4B5563",
}


def box(ax, xy, wh, title, lines, fc, ec, title_size=15, body_size=11.5):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.08",
        linewidth=2.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.28,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    for i, line in enumerate(lines):
        ax.text(
            x + 0.22,
            y + h - 0.70 - i * 0.30,
            line,
            ha="left",
            va="top",
            fontsize=body_size,
            color=COLORS["muted"],
        )
    return patch


def pill(ax, x, y, w, h, text, fc, ec, size=11, weight="bold"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color=COLORS["ink"] if ec != COLORS["red_edge"] else COLORS["red_edge"],
    )
    return patch


def arrow(ax, start, end, label=None, dy=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=2.3,
        color=COLORS["ink"],
        shrinkA=8,
        shrinkB=8,
    )
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + dy
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color=COLORS["muted"],
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
        )


def main():
    fig, ax = plt.subplots(figsize=(18, 12), dpi=240)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(
        9,
        11.56,
        "Current Pipeline: Mask-Free InstructSAM Feature Grounding for Cosmos",
        ha="center",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        9,
        11.12,
        "Goal: guide robot video generation with text-conditioned target features, without feeding an explicit target mask to Cosmos at inference.",
        ha="center",
        va="top",
        fontsize=12.2,
        color=COLORS["muted"],
    )

    # Top row: inputs -> InstructSAM -> dense target features.
    box(
        ax,
        (0.8, 8.35),
        (3.65, 2.0),
        "1. Inputs",
        ["context video frame(s)", "+ target text prompt", 'example: "yellow banana"'],
        COLORS["green"],
        COLORS["green_edge"],
    )
    box(
        ax,
        (5.4, 8.35),
        (4.15, 2.0),
        "2. InstructSAM",
        [
            "text-conditioned SAM3 decoder",
            "stage2 LoRA target checkpoint",
            "export decoder_dense only",
            "no mask is passed to Cosmos",
        ],
        COLORS["blue"],
        COLORS["blue_edge"],
        body_size=10.8,
    )
    box(
        ax,
        (10.55, 8.35),
        (4.55, 2.0),
        "3. Dense Target Feature",
        ["target_feature: [B, 1024, 256]", "1024 tokens = 32 x 32 feature map", "semantic + spatial target cues"],
        COLORS["purple"],
        COLORS["purple_edge"],
    )
    arrow(ax, (4.45, 9.35), (5.4, 9.35))
    arrow(ax, (9.55, 9.35), (10.55, 9.35))

    # Middle row: adapter detail and Cosmos.
    box(
        ax,
        (1.0, 5.24),
        (7.9, 2.15),
        "4. Target Latent Adapter",
        [],
        COLORS["gray"],
        COLORS["gray_edge"],
        title_size=14.5,
    )
    steps = [
        ("[B,1024,256]", "reshape", "[B,32,32,256]"),
        ("32x32 map", "resize", "[B,H,W,256]"),
        ("+ coords", "concat", "[B,T,H,W,259]"),
        ("MLP -> D", "project", "[B,T,H,W,D]"),
    ]
    sx = 1.48
    sy = 6.28
    for i, (top, mid, bot) in enumerate(steps):
        px = sx + i * 1.82
        pill(ax, px, sy, 1.38, 0.48, top, "#FFFFFF", COLORS["gray_edge"], size=8.9, weight="normal")
        ax.text(px + 0.69, sy - 0.22, mid, ha="center", va="center", fontsize=9.4, color=COLORS["muted"])
        ax.text(px + 0.69, sy - 0.50, bot, ha="center", va="center", fontsize=8.6, color=COLORS["ink"])
        if i < len(steps) - 1:
            arrow(ax, (px + 1.38, sy + 0.24), (px + 1.82, sy + 0.24))

    box(
        ax,
        (10.2, 5.24),
        (6.85, 2.15),
        "5. Cosmos DiT",
        [
            "inject at DiT blocks: 8, 12, 16, 20",
            "add gated feature residual to latent tokens",
            "denoise -> VAE decode -> robot video",
        ],
        COLORS["orange"],
        COLORS["orange_edge"],
        title_size=14.5,
        body_size=11.0,
    )

    arrow(ax, (12.83, 8.35), (5.0, 7.39))
    arrow(ax, (8.9, 6.4), (10.2, 6.4))

    # Formula panel.
    box(
        ax,
        (1.0, 3.36),
        (16.05, 1.28),
        "Residual Formula",
        [],
        COLORS["gray"],
        COLORS["gray_edge"],
        title_size=13.5,
    )
    ax.text(
        9.0,
        4.03,
        "x' = x + tanh(g_i) * delta_x",
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        9.0,
        3.70,
        "delta_x = MLP( concat( LayerNorm(x), target_latent_field ) )",
        ha="center",
        va="center",
        fontsize=12.0,
        color=COLORS["muted"],
    )

    # Inference and disabled notes.
    pill(
        ax,
        1.0,
        2.32,
        16.05,
        0.58,
        "Inference input = context frame(s) + target text. Cosmos receives target_feature only; no explicit target mask input.",
        COLORS["green"],
        COLORS["green_edge"],
        size=12,
        weight="normal",
    )
    pill(
        ax,
        1.0,
        1.52,
        16.05,
        0.58,
        "Disabled in this version: mask channel | target-feature cross-attention | dense token-level contrastive loss | margin loss",
        COLORS["red"],
        COLORS["red_edge"],
        size=10.8,
    )

    ax.text(
        16.95,
        0.74,
        "Experiment: latent_grounding_decoder_dense_target, 2000 steps, global batch 128",
        ha="right",
        va="center",
        fontsize=9.8,
        color="#64748B",
    )

    fig.savefig(PNG, bbox_inches="tight", facecolor="white")
    fig.savefig(PDF, bbox_inches="tight", facecolor="white")
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
