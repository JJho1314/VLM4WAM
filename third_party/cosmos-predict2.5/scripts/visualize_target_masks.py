"""Create mask-overlay visualizations for a Cosmos target-aware VideoDataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_mask(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "masks_packed" in data.files and "shape" in data.files:
        shape = tuple(int(dim) for dim in data["shape"].tolist())
        flat_pixels = int(np.prod(shape[1:]))
        return np.unpackbits(data["masks_packed"], axis=1)[:, :flat_pixels].reshape(shape).astype(bool)

    key = "masks" if "masks" in data.files else data.files[0]
    mask = data[key]
    if mask.ndim == 5:
        mask = mask.max(axis=0)
    if mask.ndim == 4:
        if mask.shape[1] == 1:
            mask = mask[:, 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask.max(axis=0)
    if mask.ndim != 3:
        raise ValueError(f"Unsupported mask shape {mask.shape} in {path}")
    return mask.astype(bool)


def read_frame(video: Path, frame_id: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame {frame_id} from {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def video_frame_count(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count <= 0:
        raise ValueError(f"bad frame count for {video}")
    return count


def resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    m = resize_mask(mask, frame.shape[0], frame.shape[1])
    red = np.zeros_like(out)
    red[..., 0] = 255
    out[m] = (0.55 * out[m] + 0.45 * red[m]).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 0), 2)
    return out


def load_frame_ranges(dataset: Path) -> dict[str, list[list[int]]]:
    path = dataset / "frame_ranges.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def choose_frame_ids(name: str, total_frames: int, frame_ranges: dict[str, list[list[int]]], frames_per_sample: int) -> list[int]:
    ranges = frame_ranges.get(name) or [[0, total_frames - 1]]
    lengths = [max(0, int(end) - int(start) + 1) for start, end in ranges]
    if not lengths or max(lengths) <= 0:
        start, end = 0, total_frames - 1
    else:
        start, end = ranges[int(np.argmax(lengths))]
        start, end = max(0, int(start)), min(total_frames - 1, int(end))
    if frames_per_sample <= 1:
        return [int(round((start + end) / 2))]
    return [int(round(x)) for x in np.linspace(start, end, frames_per_sample)]


def mask_frame_id(video_frame_id: int, total_video_frames: int, total_mask_frames: int) -> int:
    if total_video_frames <= 1 or total_mask_frames <= 1:
        return 0
    return int(np.clip(round(video_frame_id * (total_mask_frames - 1) / (total_video_frames - 1)), 0, total_mask_frames - 1))


def read_metadata(dataset: Path) -> dict[str, dict[str, str]]:
    path = dataset / "metadata.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def draw_label(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text)
    draw.rectangle([0, 0, x1 + 10, y1 + 8], fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 255))


def make_contact_sheet(items: list[tuple[str, np.ndarray]], out_path: Path, cols: int) -> None:
    thumbs = []
    for label, frame in items:
        image = Image.fromarray(frame)
        image.thumbnail((320, 180), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (320, 180), (0, 0, 0))
        canvas.paste(image, ((320 - image.width) // 2, (180 - image.height) // 2))
        draw_label(canvas, label)
        thumbs.append(canvas)

    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 320, rows * 180), (25, 25, 25))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 320, (idx // cols) * 180))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--frames-per-sample", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--names", default="", help="Comma-separated sample names to visualize.")
    args = parser.parse_args()

    dataset = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    videos = dataset / "videos"
    masks = dataset / "masks"
    frame_ranges = load_frame_ranges(dataset)
    metadata = read_metadata(dataset)

    all_names = sorted(path.stem for path in videos.glob("*.mp4") if (masks / f"{path.stem}.npz").exists())
    if args.names:
        names = [name.strip() for name in args.names.split(",") if name.strip()]
    else:
        rng = np.random.default_rng(args.seed)
        sample_count = min(args.num_samples, len(all_names))
        names = [all_names[idx] for idx in sorted(rng.choice(len(all_names), size=sample_count, replace=False))]

    all_items: list[tuple[str, np.ndarray]] = []
    summary = []
    for name in names:
        video_path = videos / f"{name}.mp4"
        mask_path = masks / f"{name}.npz"
        total_video_frames = video_frame_count(video_path)
        mask = load_mask(mask_path)
        frame_ids = choose_frame_ids(name, total_video_frames, frame_ranges, args.frames_per_sample)
        sample_items = []
        for frame_id in frame_ids:
            mid = mask_frame_id(frame_id, total_video_frames, mask.shape[0])
            frame = read_frame(video_path, frame_id)
            overlaid = overlay(frame, mask[mid])
            label = f"{name} f{frame_id}->m{mid}"
            sample_items.append((label, overlaid))
            all_items.append((label, overlaid))
        make_contact_sheet(sample_items, out_dir / f"{name}.jpg", cols=args.frames_per_sample)
        row = metadata.get(name, {})
        summary.append(
            {
                "name": name,
                "episode_index": row.get("episode_index", ""),
                "camera": row.get("camera", ""),
                "video_frames": total_video_frames,
                "mask_frames": int(mask.shape[0]),
                "frame_ids": frame_ids,
                "mask_pixels_mean": float(mask.mean()),
                "task": row.get("task_orig", ""),
            }
        )

    make_contact_sheet(all_items, out_dir / "overview.jpg", cols=args.frames_per_sample)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "samples": len(names), "overview": str(out_dir / "overview.jpg")}, indent=2))


if __name__ == "__main__":
    main()
