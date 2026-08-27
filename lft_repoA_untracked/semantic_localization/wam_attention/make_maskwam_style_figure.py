"""Render a MaskWAM-style qualitative comparison from saved paired attention maps.

The script is intentionally cheap: it reads the RGB-only and SG-WAM attention
captures produced by ``generate_many.py`` and never reruns either model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image


DEFAULT_DATA_DIR = Path("/data/LFT-W02_data/junjie/fastwam_sg_ckpt")
DEFAULT_OUTPUT = Path(
    "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/"
    "semantic_localization/figs/sgwam_maskwam_style"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--suffix", default="_main")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-scenes", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def normalized_mass(attention: np.ndarray) -> np.ndarray:
    shifted = attention.astype(np.float64) - float(attention.min())
    return shifted / (float(shifted.sum()) + 1e-9)


def concentration(attention: np.ndarray) -> float:
    mass = normalized_mass(attention).ravel()
    count = max(1, int(0.10 * mass.size))
    return float(np.sort(mass)[::-1][:count].sum())


def edge_fraction(attention: np.ndarray) -> float:
    mass = normalized_mass(attention)
    border = np.concatenate(
        (mass[0], mass[-1], mass[1:-1, 0], mass[1:-1, -1])
    )
    return float(border.sum())


def select_diverse_scenes(
    rgb_maps: np.ndarray,
    sg_maps: np.ndarray,
    prompts: list[str],
    count: int,
) -> list[dict[str, object]]:
    ranked = []
    for index, prompt in enumerate(prompts):
        rgb_concentration = concentration(rgb_maps[index])
        sg_concentration = concentration(sg_maps[index])
        score = (
            sg_concentration
            - rgb_concentration
            - 0.5 * edge_fraction(sg_maps[index])
        )
        ranked.append(
            {
                "index": index,
                "prompt": prompt,
                "score": score,
                "rgb_concentration": rgb_concentration,
                "sg_concentration": sg_concentration,
            }
        )

    selected = []
    used_prompts: set[str] = set()
    for item in sorted(ranked, key=lambda entry: float(entry["score"]), reverse=True):
        prompt = str(item["prompt"])
        if prompt in used_prompts:
            continue
        selected.append(item)
        used_prompts.add(prompt)
        if len(selected) == count:
            break

    if len(selected) != count:
        raise ValueError(
            f"requested {count} distinct scenes, but only found {len(selected)}"
        )
    return selected


def sharpen(attention: np.ndarray) -> np.ndarray:
    normalized = attention - float(attention.min())
    normalized /= float(normalized.max()) + 1e-6
    low, high = np.percentile(normalized, [60, 99])
    clipped = np.clip((normalized - low) / (high - low + 1e-6), 0.0, 1.0)
    return clipped**1.6


def overlay(frame: np.ndarray, attention: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    heat = Image.fromarray(sharpen(attention).astype(np.float32))
    heat = heat.resize((width, height), Image.Resampling.BILINEAR)
    heat_rgb = matplotlib.colormaps["turbo"](np.asarray(heat))[..., :3]
    base = frame.astype(np.float32) / 255.0
    return np.clip(0.48 * base + 0.52 * heat_rgb, 0.0, 1.0)


def add_row_container(
    fig: plt.Figure,
    *,
    y: float,
    height: float,
    color: str,
) -> None:
    container = FancyBboxPatch(
        (0.025, y),
        0.95,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.022",
        transform=fig.transFigure,
        facecolor=color,
        edgecolor="#111111",
        linewidth=1.05,
        zorder=-10,
    )
    fig.add_artist(container)


def add_panel(
    fig: plt.Figure,
    image: np.ndarray,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    shadow = Rectangle(
        (x + 0.006, y - 0.008),
        width,
        height,
        transform=fig.transFigure,
        facecolor="black",
        edgecolor="none",
        alpha=0.18,
        zorder=-1,
    )
    fig.add_artist(shadow)
    axis = fig.add_axes((x, y, width, height), zorder=2)
    axis.imshow(image, interpolation="bilinear")
    axis.set_axis_off()


def render(
    frames: np.ndarray,
    rgb_maps: np.ndarray,
    sg_maps: np.ndarray,
    selected: list[dict[str, object]],
    output: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(10.4, 6.45), facecolor="white")
    add_row_container(fig, y=0.525, height=0.435, color="#eeeeee")
    add_row_container(fig, y=0.045, height=0.435, color="#ffffe8")

    fig.text(
        0.5,
        0.922,
        "RGB-only WAM",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        family="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.442,
        "SG-WAM (Ours)",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        family="DejaVu Sans",
    )

    panel_width = 0.205
    panel_height = 0.305
    x_positions = [0.047, 0.282, 0.517, 0.752]
    top_y = 0.575
    bottom_y = 0.095
    for column, item in enumerate(selected):
        index = int(item["index"])
        add_panel(
            fig,
            overlay(frames[index], rgb_maps[index]),
            x=x_positions[column],
            y=top_y,
            width=panel_width,
            height=panel_height,
        )
        add_panel(
            fig,
            overlay(frames[index], sg_maps[index]),
            x=x_positions[column],
            y=bottom_y,
            width=panel_width,
            height=panel_height,
        )

    for column in (0, len(selected) - 1):
        center_x = x_positions[column] + panel_width / 2
        arrow = FancyArrowPatch(
            (center_x, 0.407),
            (center_x, 0.568),
            transform=fig.transFigure,
            arrowstyle="<|-|>",
            mutation_scale=17,
            linewidth=2.2,
            color="#1b1b1b",
            shrinkA=0,
            shrinkB=0,
            zorder=8,
        )
        fig.add_artist(arrow)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output.with_suffix(".png"),
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
    )
    fig.savefig(
        output.with_suffix(".pdf"),
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rgb_path = args.data_dir / f"wam_part_rgb{args.suffix}.npz"
    sg_path = args.data_dir / f"wam_part_sg{args.suffix}.npz"
    rgb_data = np.load(rgb_path, allow_pickle=True)
    sg_data = np.load(sg_path, allow_pickle=True)

    count = min(len(rgb_data["maps"]), len(sg_data["maps"]))
    rgb_maps = rgb_data["maps"][:count]
    sg_maps = sg_data["maps"][:count]
    frames = rgb_data["frames"][:count]
    prompts = [str(prompt) for prompt in rgb_data["prompts"][:count]]
    selected = select_diverse_scenes(
        rgb_maps,
        sg_maps,
        prompts,
        args.num_scenes,
    )

    render(frames, rgb_maps, sg_maps, selected, args.output, args.dpi)
    metadata = {
        "rgb_maps": str(rgb_path),
        "sg_maps": str(sg_path),
        "selection": selected,
        "outputs": [
            str(args.output.with_suffix(".png")),
            str(args.output.with_suffix(".pdf")),
        ],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for item in selected:
        print(
            f"index={item['index']:>3} score={item['score']:+.3f} "
            f"| {item['prompt']}"
        )
    print(f"saved {args.output.with_suffix('.png')}")
    print(f"saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
