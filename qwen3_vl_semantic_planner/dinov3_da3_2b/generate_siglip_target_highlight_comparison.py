"""Generate phase-aware SigLIP target-highlight palette comparisons."""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_target_highlight import (
    SiglipPairGradCAM,
)

PHASE_BOUNDARY = 128
TARGET_BEFORE = "the white textured mug"
TARGET_AFTER = "the yellow and white mug"
PALETTES = {
    "A_current": (0, 1, 2),
    "B_warm_balanced": (1, 2, 0),
    "C_cool_balanced": (2, 0, 1),
}
CAMERAS = ("main", "wrist")
FRAMES = (112, 160)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
CONTENT_SIZE = 256
COLUMN_LABEL_HEIGHT = 44
ROW_LABEL_WIDTH = 36
GUTTER = 12
QUANTILES = {"low": 0.05, "high": 0.95}
Highlighter = Callable[[Sequence[Image.Image], Sequence[str]], np.ndarray]


def active_target(frame_index: int) -> str:
    """Return the fixed active target phrase for an episode frame."""
    return TARGET_BEFORE if frame_index < PHASE_BOUNDARY else TARGET_AFTER


def permute_palette(feature_rgb: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    """Apply an exact RGB channel permutation without changing values."""
    return feature_rgb[..., order]


def _one_pixel_contour(target: np.ndarray) -> np.ndarray:
    """Return the max-pool minus min-pool boundary of a binary target."""
    padded = np.pad(target.astype(np.uint8), 1, mode="edge")
    windows = [
        padded[row : row + target.shape[0], column : column + target.shape[1]]
        for row in range(3)
        for column in range(3)
    ]
    return np.maximum.reduce(windows) - np.minimum.reduce(windows)


def combine_target_highlight(feature_rgb: np.ndarray, relevance: np.ndarray) -> np.ndarray:
    """Dim non-target regions and add a warm target fill and amber contour."""
    color = feature_rgb.astype(np.float32) / 255.0
    relevance = np.asarray(relevance, dtype=np.float32).clip(0.0, 1.0)
    grayscale = np.sum(color * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=2)
    background = 0.75 * (0.40 * color + 0.60 * grayscale[..., None])
    combined = background * (1.0 - relevance[..., None]) + color * relevance[..., None]
    warm_yellow = np.array([1.0, 0.72, 0.0], dtype=np.float32)
    combined = combined * (1.0 - 0.28 * relevance[..., None]) + warm_yellow * (
        0.28 * relevance[..., None]
    )
    contour = _one_pixel_contour(relevance >= 0.65).astype(bool)
    combined[contour] = np.array([1.0, 0.55, 0.0], dtype=np.float32)
    return np.rint(combined * 255.0).clip(0, 255).astype(np.uint8)


def _source_paths(export_root: Path) -> list[tuple[int, str, Path, Path]]:
    sources = []
    for frame_index in FRAMES:
        for camera in CAMERAS:
            frame_dir = export_root / camera / f"frame_{frame_index:06d}"
            sources.append((frame_index, camera, frame_dir / "rgb.png", frame_dir / "siglip_probe.png"))
    return sources


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    if rgb.size != (CONTENT_SIZE, CONTENT_SIZE):
        raise ValueError(f"Expected a {CONTENT_SIZE}x{CONTENT_SIZE} image at {path}")
    return rgb


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def _draw_centered_label(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text(
        ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
        text,
        fill="black",
        font=font,
        anchor="mm",
    )


def _draw_row_label(
    canvas: Image.Image,
    top: int,
    frame_index: int,
    camera: str,
    phrase: str,
) -> None:
    label = Image.new("RGB", (CONTENT_SIZE, ROW_LABEL_WIDTH), "white")
    _draw_centered_label(label, (0, 0, CONTENT_SIZE, ROW_LABEL_WIDTH), f"{frame_index} · {camera} · {phrase}", _font(11))
    canvas.paste(label.rotate(90, expand=True), (0, top))


def _canvas_size() -> tuple[int, int]:
    width = ROW_LABEL_WIDTH + 4 * CONTENT_SIZE + 3 * GUTTER
    height = COLUMN_LABEL_HEIGHT + 4 * CONTENT_SIZE + 3 * GUTTER
    return width, height


def _render_grid(
    rgb_images: Sequence[Image.Image],
    probe_images: Sequence[Image.Image],
    relevances: np.ndarray,
    row_data: Sequence[dict[str, object]],
) -> Image.Image:
    canvas = Image.new("RGB", _canvas_size(), "white")
    labels = ("RGB", "A · 当前", "B · 暖色", "C · 冷色")
    label_font = _font(18)
    for column, label in enumerate(labels):
        left = ROW_LABEL_WIDTH + column * (CONTENT_SIZE + GUTTER)
        _draw_centered_label(
            canvas,
            (left, 0, left + CONTENT_SIZE, COLUMN_LABEL_HEIGHT),
            label,
            label_font,
        )
    for row, (rgb, probe, relevance, row_info) in enumerate(
        zip(rgb_images, probe_images, relevances, row_data, strict=True)
    ):
        top = COLUMN_LABEL_HEIGHT + row * (CONTENT_SIZE + GUTTER)
        _draw_row_label(
            canvas,
            top,
            int(row_info["frame"]),
            str(row_info["camera"]),
            str(row_info["phrase"]),
        )
        canvas.paste(rgb, (ROW_LABEL_WIDTH, top))
        probe_array = np.asarray(probe)
        for column, order in enumerate(PALETTES.values(), start=1):
            panel = combine_target_highlight(permute_palette(probe_array, order), relevance)
            left = ROW_LABEL_WIDTH + column * (CONTENT_SIZE + GUTTER)
            canvas.paste(Image.fromarray(panel), (left, top))
    return canvas


def _write_atomically(
    image: Image.Image,
    metadata: dict[str, object],
    png_path: Path,
    json_path: Path,
) -> None:
    temporary_png = png_path.with_name(f".{png_path.name}.{uuid4().hex}.tmp")
    temporary_json = json_path.with_name(f".{json_path.name}.{uuid4().hex}.tmp")
    try:
        image.save(temporary_png, format="PNG")
        temporary_json.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary_png.replace(png_path)
        temporary_json.replace(json_path)
    finally:
        temporary_png.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)


def generate_comparison(
    export_root: Path,
    model_dir: Path,
    output_dir: Path,
    device: torch.device,
    *,
    highlighter: Highlighter | None = None,
) -> tuple[Path, Path]:
    """Render all fixed-phase, two-camera target-highlight palette panels."""
    export_root = Path(export_root)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    sources = _source_paths(export_root)
    missing = [path for _, _, rgb, probe in sources for path in (rgb, probe) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required export source(s): {', '.join(map(str, missing))}")

    png_path = output_dir / "siglip_target_highlight_palettes.png"
    json_path = output_dir / "siglip_target_highlight_palettes.json"
    if png_path.exists() or json_path.exists():
        raise FileExistsError(f"Refusing to overwrite {png_path} or {json_path}")

    rgb_images = [_load_rgb(rgb) for _, _, rgb, _ in sources]
    probe_images = [_load_rgb(probe) for _, _, _, probe in sources]
    row_data = [
        {"frame": frame_index, "camera": camera, "phrase": active_target(frame_index)}
        for frame_index, camera, _, _ in sources
    ]
    phrases = [str(row["phrase"]) for row in row_data]
    if highlighter is None:
        highlighter = SiglipPairGradCAM(model_dir, device)
    relevances = np.asarray(highlighter(rgb_images, phrases), dtype=np.float32)
    expected_shape = (len(sources), CONTENT_SIZE, CONTENT_SIZE)
    if relevances.shape != expected_shape:
        raise ValueError(f"Expected relevance maps shaped {expected_shape}, got {relevances.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "model_path": str(model_dir),
        "phase_boundary": PHASE_BOUNDARY,
        "frames": list(FRAMES),
        "cameras": list(CAMERAS),
        "phrases": phrases,
        "quantiles": QUANTILES,
        "palettes": {name: list(order) for name, order in PALETTES.items()},
        "panel_order": row_data,
    }
    _write_atomically(_render_grid(rgb_images, probe_images, relevances, row_data), metadata, png_path, json_path)
    return png_path, json_path


def main() -> None:
    """Run the fixed-phase palette comparison generator from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--siglip2-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    generate_comparison(
        args.export_root,
        args.siglip2_model_dir,
        args.output_dir,
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
