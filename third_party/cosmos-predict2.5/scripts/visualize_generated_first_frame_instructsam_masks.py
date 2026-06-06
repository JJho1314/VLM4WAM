#!/usr/bin/env python3
"""Run InstructSAM on generated first frames and compare with attention maps."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont

from cosmos_predict2._src.predict2.target_aware.instructsam_mask import InstructSAMTargetMaskGenerator


STOP_WORDS = {
    "and",
    "from",
    "in",
    "inside",
    "into",
    "of",
    "off",
    "on",
    "onto",
    "out",
    "over",
    "then",
    "to",
    "under",
    "with",
    "backward",
    "backwards",
    "forward",
    "forwards",
}


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def extract_phrase(caption: str) -> str:
    if "[TGT]" not in caption:
        return "target object"
    tail = caption.split("[TGT]", 1)[1].strip()
    tail = re.split(r"[,.;:!?]", tail, maxsplit=1)[0].strip()
    kept: list[str] = []
    for token in tail.split():
        clean = token.strip("\"'`()[]{}").lower()
        if kept and clean in STOP_WORDS:
            break
        kept.append(token.strip("\"'`()[]{}"))
    phrase = " ".join(part for part in kept if part).strip()
    if phrase.lower().startswith(("the ", "a ", "an ")):
        phrase = " ".join(phrase.split()[1:])
    return phrase or "target object"


def read_video_frame(path: Path, index: int) -> Image.Image:
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    nframes = len(vr)
    index = max(0, min(index, nframes - 1))
    frame = vr.get_batch([index]).asnumpy()[0]
    return Image.fromarray(frame).convert("RGB")


def read_video_triplet(path: Path) -> tuple[list[Image.Image], list[int]]:
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    nframes = len(vr)
    indices = [0, max(0, nframes // 2), max(0, nframes - 1)]
    frames = [Image.fromarray(vr.get_batch([idx]).asnumpy()[0]).convert("RGB") for idx in indices]
    return frames, indices


def read_video_grid_frames(path: Path) -> tuple[list[Image.Image], list[int]]:
    vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    nframes = len(vr)
    indices = sorted(set([0, nframes // 4, nframes // 2, 3 * nframes // 4, nframes - 1]))
    frames = [Image.fromarray(vr.get_batch([idx]).asnumpy()[0]).convert("RGB") for idx in indices]
    return frames, indices


def mask_array(result) -> np.ndarray:
    return result.mask_B_C_T_H_W[0, 0, 0].detach().float().cpu().numpy() > 0


def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 80, 20), alpha=0.45) -> Image.Image:
    base = image.convert("RGBA")
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(base.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", base.size, (*color, 0))
    alpha_arr = np.asarray(mask_img).astype(np.float32) / 255.0
    alpha_img = Image.fromarray((alpha_arr * 255 * alpha).astype(np.uint8), mode="L")
    overlay.putalpha(alpha_img)
    return Image.alpha_composite(base, overlay).convert("RGB")


def bbox_and_centroid(mask: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int] | None]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    centroid = (int(xs.mean()), int(ys.mean()))
    return bbox, centroid


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont | None = None) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def wrap_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    max_width: int,
    font: ImageFont.ImageFont | None = None,
    max_lines: int = 2,
) -> list[str]:
    words = label.split()
    if not words:
        return [label]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if len(lines) < max_lines:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if len(lines) == max_lines and text_width(draw, lines[-1], font) > max_width:
        line = lines[-1]
        while len(line) > 4 and text_width(draw, f"{line}...", font) > max_width:
            line = line[:-1]
        lines[-1] = f"{line}..."
    return lines


def label_image(
    image: Image.Image,
    label: str,
    font: ImageFont.ImageFont | None = None,
    label_height: int = 56,
) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, label_height), fill=(0, 0, 0))
    lines = wrap_label(draw, label, max(out.width - 16, 32), font=font, max_lines=2)
    y = 8
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255), font=font)
        bbox = draw.textbbox((8, y), line, font=font)
        y += max(18, bbox[3] - bbox[1] + 4)
    return out


def draw_mask_panel(
    image: Image.Image,
    mask: np.ndarray,
    label: str,
    font: ImageFont.ImageFont | None = None,
    label_height: int = 56,
) -> Image.Image:
    out = overlay_mask(image, mask)
    draw = ImageDraw.Draw(out)
    bbox, centroid = bbox_and_centroid(mask)
    if bbox is not None:
        draw.rectangle(bbox, outline=(0, 255, 255), width=4)
    if centroid is not None:
        cx, cy = centroid
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=(0, 255, 0), width=4)
    return label_image(out, label, font, label_height=label_height)


def resize_width(image: Image.Image, width: int) -> Image.Image:
    height = int(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.BILINEAR)


def make_grid(
    rows: list[list[Image.Image]],
    tile_w: int = 320,
    column_widths: list[int] | None = None,
) -> Image.Image:
    ncols = max(len(row) for row in rows)
    if column_widths is None:
        column_widths = [tile_w] * ncols
    if len(column_widths) < ncols:
        column_widths = column_widths + [column_widths[-1]] * (ncols - len(column_widths))

    thumbs: list[list[Image.Image]] = []
    row_heights: list[int] = []
    for row in rows:
        resized = [resize_width(img, column_widths[c]) for c, img in enumerate(row)]
        thumbs.append(resized)
        row_heights.append(max(img.height for img in resized))

    canvas = Image.new("RGB", (sum(column_widths[:ncols]), sum(row_heights)), "white")
    y = 0
    for r, row in enumerate(thumbs):
        x = 0
        for c, img in enumerate(row):
            canvas.paste(img, (x, y))
            x += column_widths[c]
        y += row_heights[r]
    return canvas


def stack_images(images: list[Image.Image], gap: int = 24, background: str = "white") -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), background)
    width = max(img.width for img in images)
    height = sum(img.height for img in images) + gap * max(0, len(images) - 1)
    canvas = Image.new("RGB", (width, height), background)
    y = 0
    for image in images:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    return canvas


def resize_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height)
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def resize_fill(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized_width = max(width, int(image.width * scale + 0.5))
    resized_height = max(height, int(image.height * scale + 0.5))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = max(0, (resized_width - width) // 2)
    top = max(0, (resized_height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def make_projection_row(
    frames: list[Image.Image],
    mask: np.ndarray,
    total_width: int,
    label: str,
    side_font: ImageFont.ImageFont | None,
    label_width: int = 360,
    row_height: int = 360,
) -> Image.Image:
    label_width = min(label_width, max(180, total_width // 3))
    content_width = max(1, total_width - label_width)
    ncols = max(1, len(frames))
    canvas = Image.new("RGB", (total_width, row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, label_width, row_height), fill=(245, 245, 245))
    y = 18
    for line in wrap_label(draw, label, max(label_width - 28, 32), font=side_font, max_lines=4):
        draw.text((14, y), line, fill=(0, 0, 0), font=side_font)
        bbox = draw.textbbox((14, y), line, font=side_font)
        y += max(28, bbox[3] - bbox[1] + 8)

    for idx, frame in enumerate(frames):
        col_left = label_width + idx * content_width // ncols
        col_right = label_width + (idx + 1) * content_width // ncols
        tile_width = max(1, col_right - col_left)
        overlay = overlay_mask(frame, mask, color=(255, 0, 255), alpha=0.72)
        draw = ImageDraw.Draw(overlay)
        bbox, centroid = bbox_and_centroid(mask)
        if bbox is not None:
            draw.rectangle(bbox, outline=(0, 255, 255), width=6)
        if centroid is not None:
            cx, cy = centroid
            draw.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), outline=(0, 255, 0), width=5)
        tile = resize_fill(overlay, tile_width, row_height)
        canvas.paste(tile, (col_left, 0))
    return canvas


def insert_projection_after_target_mask(
    attention_panel: Image.Image,
    projection_row: Image.Image,
    header_height: int = 48,
    rows_before_insert: int = 2,
    original_rows: int = 3,
) -> Image.Image:
    if original_rows <= 0 or attention_panel.height <= header_height:
        return stack_images([projection_row, attention_panel], gap=0)
    row_height = max(1, (attention_panel.height - header_height) // original_rows)
    insert_y = min(attention_panel.height, header_height + rows_before_insert * row_height)
    projection = projection_row.resize((attention_panel.width, row_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (attention_panel.width, attention_panel.height + row_height), "white")
    canvas.paste(attention_panel.crop((0, 0, attention_panel.width, insert_y)), (0, 0))
    canvas.paste(projection, (0, insert_y))
    canvas.paste(
        attention_panel.crop((0, insert_y, attention_panel.width, attention_panel.height)),
        (0, insert_y + row_height),
    )
    return canvas


def make_wrapped_sample_sheet(
    top_panels: list[Image.Image],
    attention_panel: Image.Image,
    tile_w: int,
    gap: int = 12,
) -> Image.Image:
    top_grid = make_grid([top_panels], tile_w=tile_w)
    attention = resize_width(attention_panel, top_grid.width)
    return stack_images([top_grid, attention], gap=gap)


def load_b16_metrics(summary_path: Path) -> dict[int, dict[str, float]]:
    if not summary_path.exists():
        return {}
    data = json.load(open(summary_path))
    out: dict[int, dict[str, float]] = {}
    for sample in data.get("samples", []):
        b16 = next((m for m in sample.get("block_metrics", []) if m.get("block") == 16), None)
        if b16:
            out[int(sample["sample_index"])] = {
                "ratio": float(b16["inside_outside_ratio"]),
                "mass": float(b16["attn_mass_inside_mask"]),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--attention-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--combine-mode", choices=["best", "union"], default="best")
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--sample-tile-width", type=int, default=480)
    parser.add_argument("--contact-tile-width", type=int, default=420)
    parser.add_argument("--font-size", type=int, default=26)
    parser.add_argument("--side-label-font-size", type=int, default=38)
    parser.add_argument("--label-height", type=int, default=72)
    parser.add_argument("--projection-label-width", type=int, default=360)
    parser.add_argument("--projection-row-height", type=int, default=360)
    parser.add_argument("--row-gap", type=int, default=16)
    parser.add_argument("--sample-gap", type=int, default=36)
    args = parser.parse_args()

    run_root = args.run_root
    output_dir = args.output_dir or (run_root / "generated_first_frame_instructsam_check")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", args.font_size)
        side_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", args.side_label_font_size)
    except Exception:
        font = None
        side_font = None

    device_map: str | dict[str, str] = {"": "cuda:0"} if torch.cuda.is_available() else "cpu"
    generator = InstructSAMTargetMaskGenerator(
        args.model_path,
        source_root=args.source_root,
        device_map=device_map,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
    )

    attention_dir = args.attention_dir or (run_root / "instructsam_feature_attention")
    text_metrics = load_b16_metrics(run_root / "text_target_attention" / "cross_attention_visualization_summary.json")
    feature_metrics = load_b16_metrics(attention_dir / "cross_attention_visualization_summary.json")

    generated_dir = run_root / "generation"
    sample_paths = sorted(generated_dir.glob("sample_*_generated.mp4"))
    if not sample_paths:
        generated_dir = run_root
        sample_paths = sorted(generated_dir.glob("sample_*_generated.mp4"))
    if args.limit > 0:
        sample_paths = sample_paths[: args.limit]

    contact_blocks = []
    records = []
    for gen_path in sample_paths:
        sample_index = int(gen_path.stem.split("_")[1])
        caption = (generated_dir / f"sample_{sample_index:03d}_caption.txt").read_text().strip()
        phrase = extract_phrase(caption)
        query = f"Please segment '{phrase}' in the image."
        frames, _ = read_video_triplet(gen_path)
        projection_frames, _ = read_video_grid_frames(gen_path)
        first, mid, last = frames

        result = generator.predict(
            first,
            query,
            combine_mode=args.combine_mode,
            mask_threshold=args.mask_threshold,
            output_size=(first.height, first.width),
            feature_mode="mask_query",
        )
        mask = mask_array(result)
        mask_png = output_dir / f"sample_{sample_index:03d}_generated_first_instructsam_mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_png)

        effect_path = attention_dir / f"sample_{sample_index:03d}_effect_cross_attention_loss.jpg"
        effect = Image.open(effect_path).convert("RGB") if effect_path.exists() else Image.new("RGB", first.size, "white")

        top_panels = [
            label_image(first, f"{sample_index:03d} first | {phrase}", font, label_height=args.label_height),
            draw_mask_panel(
                first,
                mask,
                f"InstructSAM first-frame mask score={result.score:.4f}",
                font,
                label_height=args.label_height,
            ),
            label_image(mid, "generated mid frame", font, label_height=args.label_height),
            label_image(last, "generated last frame", font, label_height=args.label_height),
        ]
        attention_panel = label_image(
            effect,
            "Cosmos feature-attn visualization",
            font,
            label_height=args.label_height,
        )
        projection_row = make_projection_row(
            projection_frames,
            mask,
            total_width=attention_panel.width,
            label="InstructSAM mask projection",
            side_font=side_font,
            label_width=args.projection_label_width,
            row_height=args.projection_row_height,
        )
        attention_panel = insert_projection_after_target_mask(attention_panel, projection_row)
        sample_sheet = make_wrapped_sample_sheet(
            top_panels,
            attention_panel,
            tile_w=args.sample_tile_width,
            gap=args.row_gap,
        )
        sample_sheet_path = output_dir / f"sample_{sample_index:03d}_generated_first_mask_vs_video.jpg"
        sample_sheet.save(sample_sheet_path, quality=98, subsampling=0)

        contact_block = make_wrapped_sample_sheet(
            top_panels,
            attention_panel,
            tile_w=args.contact_tile_width,
            gap=args.row_gap,
        )
        contact_blocks.append(contact_block)

        bbox, centroid = bbox_and_centroid(mask)
        record = {
            "sample_index": sample_index,
            "caption": caption,
            "phrase": phrase,
            "query": query,
            "instructsam_text": result.text,
            "instructsam_score": result.score,
            "mask_pixels": int(mask.sum()),
            "mask_occupancy": float(mask.mean()),
            "mask_bbox": bbox,
            "mask_centroid": centroid,
            "feature_b16": feature_metrics.get(sample_index),
            "text_b16": text_metrics.get(sample_index),
            "mask_png": str(mask_png),
            "sample_sheet": str(sample_sheet_path),
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    contact_sheet = output_dir / "generated_first_frame_instructsam_mask_contact.jpg"
    stack_images(contact_blocks, gap=args.sample_gap).save(contact_sheet, quality=98, subsampling=0)
    summary = {
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "contact_sheet": str(contact_sheet),
        "num_samples": len(records),
        "records": records,
        "note": "seg_output_embeddings are target query embeddings, not directly invertible into a spatial mask without InstructSAM image features/mask decoder; this check reruns InstructSAM on generated first frames.",
    }
    (output_dir / "generated_first_frame_instructsam_mask_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
