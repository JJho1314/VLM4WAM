# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic video dataset loader for Cosmos Predict2."""

import json
import os
import random
import traceback
import glob
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from decord import VideoReader, cpu
from megatron.core import parallel_state
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo
from cosmos_predict2._src.predict2.networks.semantic_plan_conditioning import (
    load_semantic_plan_tensor,
    normalize_semantic_plan_tensor,
    semantic_plan_times_from_frame_indices,
)


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        prompt_type: str | None = None,  # "long", "short", "medium", or None for auto
        caption_format: str = "auto",  # "text", "json", or "auto"
        video_paths: Optional[list[str]] = None,
        semantic_plan_dir: Optional[str] = None,
        semantic_plan_default_to_zero: bool = False,
        semantic_plan_dim: int = 1152,
        semantic_plan_max_tokens: int = 0,
        semantic_plan_manifest: Optional[str] = None,
        semantic_plan_skip_features: bool = False,
        vae_latent_dir: Optional[str] = None,
        vae_latent_manifest: Optional[str] = None,
        episode_sampling: bool = False,
        frame_ranges_path: Optional[str] = None,
        episode_frame_strides: Optional[Sequence[int]] = None,
        episode_samples_per_epoch: int = 0,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (tuple[int, int]): Target size (H,W) for video frames
            prompt_type (str | None): Which prompt to use from JSON ("long", "short", "medium").
                                     If None, uses the first available prompt type.
                                     Only applicable when using JSON format.
            caption_format (str): Caption format - "text", "json", or "auto" to detect automatically

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames
        self.prompt_type = prompt_type
        self.caption_format = caption_format
        self.semantic_plan_dir = self._resolve_optional_dir(semantic_plan_dir)
        self.semantic_plan_default_to_zero = bool(semantic_plan_default_to_zero)
        self.semantic_plan_dim = int(semantic_plan_dim)
        self.semantic_plan_max_tokens = int(semantic_plan_max_tokens)
        self.semantic_plan_manifest = semantic_plan_manifest
        # Manifest windows/VAE latents are still used, but semantic-plan .pt features are not
        # loaded — the model encodes plans online from the video (semantic_plan_online_*).
        self.semantic_plan_skip_features = bool(semantic_plan_skip_features)
        self.vae_latent_dir = self._resolve_optional_dir(vae_latent_dir)
        self.vae_latent_manifest = vae_latent_manifest
        if self.semantic_plan_dim <= 0:
            raise ValueError("semantic_plan_dim must be positive")
        if self.semantic_plan_max_tokens < 0:
            raise ValueError("semantic_plan_max_tokens must be non-negative")

        # Determine caption format and directory
        self._setup_caption_format()

        video_dir = os.path.join(self.dataset_dir, "videos")

        if video_paths is None:
            self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
            self.video_paths = sorted(self.video_paths)
        else:
            self.video_paths = video_paths
        # Ctrl-World-style episode sampling: a per-stem frame-range index, with windows
        # constructed fully at random inside __getitem__ (random stride, random start).
        # No window manifest / precomputed features are used in this mode.
        self.episode_sampling = bool(episode_sampling)
        self.episode_frame_strides = sorted({int(s) for s in (episode_frame_strides or (1,)) if int(s) > 0})
        self.episode_samples_per_epoch = int(episode_samples_per_epoch)
        self.episode_ranges: list[tuple[str, int, int]] = []
        self.episode_range_probs: Optional[np.ndarray] = None
        if self.episode_sampling:
            self._build_episode_ranges(frame_ranges_path)

        self.semantic_plan_records = (
            [] if self.episode_sampling else self._load_semantic_plan_manifest(self.semantic_plan_manifest)
        )
        self.vae_latent_records = {} if self.episode_sampling else self._load_vae_latent_manifest(self.vae_latent_manifest)
        if self.episode_sampling:
            log.info(
                f"episode sampling: {len(self.episode_ranges)} motion ranges, "
                f"{self.episode_samples_per_epoch} samples per epoch, strides {self.episode_frame_strides}"
            )
        elif self.semantic_plan_records:
            log.info(f"{len(self.semantic_plan_records)} semantic-plan window samples in total")
        else:
            log.info(f"{len(self.video_paths)} videos in total")
        if self.vae_latent_records:
            log.info(f"{len(self.vae_latent_records)} VAE latent window samples in total")

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])

    def _build_episode_ranges(self, frame_ranges_path: Optional[str]) -> None:
        if not frame_ranges_path:
            frame_ranges_path = os.path.join(self.dataset_dir, "frame_ranges.json")
        if not os.path.isabs(frame_ranges_path):
            frame_ranges_path = os.path.join(self.dataset_dir, frame_ranges_path)
        if not os.path.exists(frame_ranges_path):
            raise FileNotFoundError(f"frame_ranges file not found for episode sampling: {frame_ranges_path}")
        with open(frame_ranges_path, "r", encoding="utf-8") as f:
            ranges_by_stem = json.load(f)
        min_span = self.sequence_length  # stride-1 window must fit
        spans: list[int] = []
        for stem, ranges in ranges_by_stem.items():
            for range_start, range_end in ranges:
                range_start = max(int(range_start), 0)
                range_end = int(range_end)
                if range_end - range_start < min_span:
                    continue
                self.episode_ranges.append((str(stem), range_start, range_end))
                spans.append(range_end - range_start)
        if not self.episode_ranges:
            raise ValueError(f"No usable frame ranges (>= {min_span} frames) in {frame_ranges_path}")
        # Per-window-equivalent weighting: ranges are drawn proportional to their length,
        # like Ctrl-World's dense per-start-frame enumeration.
        probs = np.asarray(spans, dtype=np.float64)
        self.episode_range_probs = probs / probs.sum()
        if self.episode_samples_per_epoch <= 0:
            self.episode_samples_per_epoch = max(int(sum(spans) // self.sequence_length), 1)

    def _getitem_episode(self) -> dict[str, Any]:
        ridx = int(np.random.choice(len(self.episode_ranges), p=self.episode_range_probs))
        stem, range_start, range_end = self.episode_ranges[ridx]
        strides = [
            s for s in self.episode_frame_strides if (self.sequence_length - 1) * s + 1 <= range_end - range_start
        ]
        if strides:
            # Weight strides by their number of valid start positions so that (stride, start)
            # is uniform over all valid windows of this range — matching the distribution of a
            # dense per-window enumeration instead of over-representing long strides.
            stride_weights = [
                range_end - range_start - ((self.sequence_length - 1) * s + 1) + 1 for s in strides
            ]
            frame_stride = random.choices(strides, weights=stride_weights, k=1)[0]
        else:
            frame_stride = 1
        span = (self.sequence_length - 1) * frame_stride + 1
        start = random.randint(range_start, range_end - span)
        frame_ids = [start + frame_stride * i for i in range(self.sequence_length)]

        video, fps = self._get_frames_by_ids(self._video_path_for_stem(stem), frame_ids)
        fps = float(fps) / frame_stride
        video = video.permute(1, 0, 2, 3)

        data: dict[str, Any] = {}
        data["video"] = video
        data["ai_caption"] = self._load_caption_for_stem(stem)
        _, _, h, w = video.shape
        data["fps"] = fps
        data["image_size"] = torch.tensor([h, w, h, w])
        data["num_frames"] = self.sequence_length
        data["padding_mask"] = torch.zeros(1, h, w)
        return data

    def __str__(self) -> str:
        if self.episode_sampling:
            return (
                f"episode sampling over {len(self.episode_ranges)} ranges "
                f"({self.episode_samples_per_epoch} samples/epoch) from {self.dataset_dir}"
            )
        if self.semantic_plan_records:
            return f"{len(self.semantic_plan_records)} semantic-plan window samples from {self.dataset_dir}"
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        if self.episode_sampling:
            return self.episode_samples_per_epoch
        if self.semantic_plan_records:
            return len(self.semantic_plan_records)
        return len(self.video_paths)

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # randomly sample a sequence of frames
        max_start_idx = total_frames - self.sequence_length
        start_frame = np.random.randint(0, max_start_idx + 1)
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps

    def _load_video_frame_ids(self, video_path: str, frame_ids: list[int]) -> tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if not frame_ids:
            raise ValueError(f"No frame ids provided for {video_path}")
        safe_frame_ids = [min(max(int(frame_id), 0), total_frames - 1) for frame_id in frame_ids]
        frame_data = vr.get_batch(safe_frame_ids).asnumpy()
        vr.seek(0)

        try:
            fps = vr.get_avg_fps()
        except Exception:
            fps = 16
        del vr
        return frame_data, fps

    def _resolve_optional_dir(self, path: Optional[str]) -> Optional[str]:
        if path is None or str(path).lower() == "none":
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(self.dataset_dir, path)

    def _resolve_manifest_paths(self, manifest: Optional[str]) -> list[str]:
        if not manifest or str(manifest).lower() == "none":
            return []
        candidates: list[str] = []
        for item in str(manifest).split(","):
            item = item.strip()
            if not item:
                continue
            if os.path.isabs(item):
                pattern = item
            elif self.semantic_plan_dir is not None:
                pattern = os.path.join(self.semantic_plan_dir, item)
            else:
                pattern = os.path.join(self.dataset_dir, item)
            if any(char in pattern for char in "*?[]"):
                candidates.extend(sorted(glob.glob(pattern)))
            else:
                candidates.append(pattern)
        return candidates

    def _resolve_manifest_paths_in_dir(self, manifest: Optional[str], base_dir: Optional[str]) -> list[str]:
        if not manifest or str(manifest).lower() == "none" or base_dir is None:
            return []
        candidates: list[str] = []
        for item in str(manifest).split(","):
            item = item.strip()
            if not item:
                continue
            pattern = item if os.path.isabs(item) else os.path.join(base_dir, item)
            if any(char in pattern for char in "*?[]"):
                candidates.extend(sorted(glob.glob(pattern)))
            else:
                candidates.append(pattern)
        return candidates

    def _load_semantic_plan_manifest(self, manifest: Optional[str]) -> list[dict[str, Any]]:
        manifest_paths = self._resolve_manifest_paths(manifest)
        if manifest and str(manifest).lower() != "none" and not manifest_paths:
            raise FileNotFoundError(f"Semantic plan manifest not found: {manifest}")
        if not manifest_paths:
            return []
        records: list[dict[str, Any]] = []
        for manifest_path in manifest_paths:
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"Semantic plan manifest not found: {manifest_path}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "sample_id" not in record:
                        raise ValueError(f"{manifest_path}:{line_no} missing sample_id")
                    if "stem" not in record:
                        record["stem"] = str(record["sample_id"]).split("__", 1)[0]
                    records.append(record)
        if not records:
            raise ValueError(f"No semantic plan records found in {manifest_paths}")
        return records

    def _load_vae_latent_manifest(self, manifest: Optional[str]) -> dict[str, dict[str, Any]]:
        manifest_paths = self._resolve_manifest_paths_in_dir(manifest, self.vae_latent_dir)
        if manifest and str(manifest).lower() != "none" and self.vae_latent_dir is not None and not manifest_paths:
            raise FileNotFoundError(f"VAE latent manifest not found: {self.vae_latent_dir}/{manifest}")
        records: dict[str, dict[str, Any]] = {}
        for manifest_path in manifest_paths:
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f"VAE latent manifest not found: {manifest_path}")
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    sample_id = record.get("sample_id")
                    latent_path = record.get("latent_path")
                    if not sample_id or not latent_path:
                        raise ValueError(f"{manifest_path}:{line_no} missing sample_id or latent_path")
                    records[str(sample_id)] = record
        return records

    def _setup_caption_format(self) -> None:
        """Determine the caption format and set up the caption directory."""
        metas_dir = os.path.join(self.dataset_dir, "metas")
        captions_dir = os.path.join(self.dataset_dir, "captions")

        if self.caption_format == "auto":
            # Auto-detect based on directory existence
            if os.path.exists(captions_dir) and any(f.endswith(".json") for f in os.listdir(captions_dir)):
                self.caption_format = "json"
                self.caption_dir = captions_dir
            elif os.path.exists(metas_dir) and any(f.endswith(".txt") for f in os.listdir(metas_dir)):
                self.caption_format = "text"
                self.caption_dir = metas_dir
            else:
                raise ValueError(
                    f"Could not auto-detect caption format. Neither 'metas/*.txt' nor 'captions/*.json' found in {self.dataset_dir}"
                )
        elif self.caption_format == "json":
            if not os.path.exists(captions_dir):
                raise ValueError(f"JSON format specified but 'captions' directory not found in {self.dataset_dir}")
            self.caption_dir = captions_dir
        elif self.caption_format == "text":
            if not os.path.exists(metas_dir):
                raise ValueError(f"Text format specified but 'metas' directory not found in {self.dataset_dir}")
            self.caption_dir = metas_dir
        else:
            raise ValueError(f"Invalid caption_format: {self.caption_format}. Must be 'text', 'json', or 'auto'")

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            log.warning(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _load_json_caption(self, json_path: Path) -> str:
        """Load caption from JSON file with prompt type selection."""
        try:
            with open(json_path, "r") as f:
                content = f.read()
                # Handle JSON that might not have top-level object
                if not content.strip().startswith("{"):
                    # Wrap in object if needed
                    data = json.loads("{" + content + "}")
                else:
                    data = json.loads(content)

            # Get the first model's captions (e.g., "qwen3_vl_30b_a3b")
            model_key = next(iter(data.keys()))
            captions = data[model_key]

            if self.prompt_type:
                # Use specified prompt type
                if self.prompt_type in captions:
                    return captions[self.prompt_type]
                else:
                    log.warning(
                        f"Prompt type '{self.prompt_type}' not found in {json_path}. "
                        f"Available: {list(captions.keys())}. Using first available."
                    )

            # Use first available prompt type
            first_prompt = next(iter(captions.values()))
            return first_prompt

        except Exception as e:
            log.warning(f"Failed to read JSON caption file {json_path}: {e}")
            return ""

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float]:
        frames, fps = self._load_video(video_path)
        return self._frames_to_tensor(frames), fps

    def _get_frames_by_ids(self, video_path: str, frame_ids: list[int]) -> tuple[torch.Tensor, float]:
        frames, fps = self._load_video_frame_ids(video_path, frame_ids)
        return self._frames_to_tensor(frames), fps

    def _frames_to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _semantic_plan_candidates(self, video_basename: str) -> list[str]:
        if self.semantic_plan_dir is None:
            return []
        return [
            os.path.join(self.semantic_plan_dir, f"{video_basename}{suffix}")
            for suffix in (".pt", ".pth", ".npy", ".npz")
        ]

    def _zero_semantic_plan(self) -> torch.Tensor:
        tokens = self.semantic_plan_max_tokens if self.semantic_plan_max_tokens > 0 else 1
        return torch.zeros(tokens, self.semantic_plan_dim, dtype=torch.float32)

    def _normalize_semantic_plan(self, semantic_plan: Any) -> torch.Tensor:
        return normalize_semantic_plan_tensor(
            semantic_plan,
            semantic_plan_dim=self.semantic_plan_dim,
            max_tokens=self.semantic_plan_max_tokens,
        )

    def _load_semantic_plan(self, semantic_plan_id: str) -> torch.Tensor:
        candidates = self._semantic_plan_candidates(semantic_plan_id)
        path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
        if path is None:
            if self.semantic_plan_default_to_zero:
                return self._zero_semantic_plan()
            raise FileNotFoundError(
                f"Semantic plan not found for {semantic_plan_id}; tried {candidates}"
            )
        return load_semantic_plan_tensor(
            path,
            semantic_plan_dim=self.semantic_plan_dim,
            max_tokens=self.semantic_plan_max_tokens,
        )

    def _load_vae_latent(self, sample_id: str) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.vae_latent_dir is None:
            raise RuntimeError("vae_latent_dir is not configured")
        record = self.vae_latent_records[sample_id]
        latent_path = str(record["latent_path"])
        if not os.path.isabs(latent_path):
            latent_path = os.path.join(self.vae_latent_dir, latent_path)
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        latent = payload["latent"] if isinstance(payload, dict) and "latent" in payload else payload
        if latent.ndim == 5 and latent.shape[0] == 1:
            latent = latent[0]
        if latent.ndim != 4:
            raise ValueError(f"Expected VAE latent [C,T,H,W], got {tuple(latent.shape)} from {latent_path}")
        return latent.contiguous(), record

    def _video_path_for_stem(self, stem: str) -> str:
        return os.path.join(self.dataset_dir, "videos", f"{stem}.mp4")

    def _frame_ids_from_record(self, record: dict[str, Any]) -> list[int]:
        frame_ids = record.get("video_frame_indices")
        if frame_ids:
            frame_ids = [int(frame_id) for frame_id in frame_ids]
        else:
            start = int(record.get("range_start", record.get("start", 0)))
            frame_stride = int(record.get("frame_stride", 1) or 1)
            frame_ids = [start + frame_stride * i for i in range(self.sequence_length)]
        if len(frame_ids) > self.sequence_length:
            # Longer manifest windows are cropped to sequence_length at a random offset
            # (e.g. training 49-frame clips from 93-frame window manifests). A fixed front
            # crop would leave the final (len - seq_len) frames of every episode range
            # uncovered as generation targets; the random offset restores coverage and adds
            # temporal augmentation while keeping the per-record frame stride.
            offset = random.randint(0, len(frame_ids) - self.sequence_length)
            frame_ids = frame_ids[offset : offset + self.sequence_length]
        if len(frame_ids) != self.sequence_length:
            raise ValueError(
                f"Manifest sample {record.get('sample_id')} has {len(frame_ids)} frames, "
                f"expected {self.sequence_length}"
            )
        return frame_ids

    def _frame_stride_from_record(self, record: dict[str, Any], frame_ids: list[int]) -> float:
        if record.get("frame_stride") is not None:
            return max(float(record.get("frame_stride") or 1.0), 1.0)
        if len(frame_ids) < 2:
            return 1.0
        diffs = np.diff(np.asarray(frame_ids, dtype=np.float32))
        positive_diffs = diffs[diffs > 0]
        if positive_diffs.size == 0:
            return 1.0
        return max(float(np.median(positive_diffs)), 1.0)

    def _load_caption_for_stem(self, stem: str) -> str:
        if self.caption_format == "json":
            caption_path = os.path.join(self.caption_dir, f"{stem}.json")
            return self._load_json_caption(Path(caption_path))
        caption_path = os.path.join(self.caption_dir, f"{stem}.txt")
        return self._load_text(Path(caption_path))

    def _getitem_manifest(self, index: int) -> dict[str, Any]:
        record = dict(self.semantic_plan_records[index])
        stem = str(record["stem"])
        sample_id = str(record["sample_id"])
        frame_ids = self._frame_ids_from_record(record)
        frame_stride = self._frame_stride_from_record(record, frame_ids)
        video, fps = self._get_frames_by_ids(self._video_path_for_stem(stem), frame_ids)
        fps = float(fps) / frame_stride
        video = video.permute(1, 0, 2, 3)

        data: dict[str, Any] = {}
        data["video"] = video
        data["ai_caption"] = self._load_caption_for_stem(stem)

        _, _, h, w = video.shape
        data["fps"] = fps
        data["image_size"] = torch.tensor([h, w, h, w])
        data["num_frames"] = self.sequence_length
        data["padding_mask"] = torch.zeros(1, h, w)
        if self.semantic_plan_dir is not None:
            if not self.semantic_plan_skip_features:
                data["semantic_plan"] = self._load_semantic_plan(sample_id)
                semantic_plan_times = semantic_plan_times_from_frame_indices(
                    record.get("future_frame_indices"),
                    record.get("video_frame_indices"),
                )
                if semantic_plan_times is not None:
                    data["semantic_plan_times"] = semantic_plan_times
            data["semantic_plan_meta"] = record
        if sample_id in self.vae_latent_records:
            data["vae_latent"], data["vae_latent_meta"] = self._load_vae_latent(sample_id)
        return data

    def __getitem__(self, index: int) -> dict | Any:
        try:
            if self.episode_sampling:
                return self._getitem_episode()
            if self.semantic_plan_records:
                return self._getitem_manifest(index)

            data = dict()
            video, fps = self._get_frames(self.video_paths[index])
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]

            # Load caption based on format
            video_path = self.video_paths[index]
            video_basename = os.path.basename(video_path).replace(".mp4", "")

            caption = self._load_caption_for_stem(video_basename)

            data["video"] = video
            data["ai_caption"] = caption

            _, _, h, w = video.shape

            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)
            if self.semantic_plan_dir is not None:
                data["semantic_plan"] = self._load_semantic_plan(video_basename)

            return data
        except Exception as e:
            self.num_failed_loads += 1
            if self.semantic_plan_records and index < len(self.semantic_plan_records):
                failed_item = self.semantic_plan_records[index].get("sample_id")
            elif index < len(self.video_paths):
                failed_item = self.video_paths[index]
            else:
                failed_item = f"episode-sample-{index}"
            log.warning(
                f"Failed to load video {failed_item} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            if self.semantic_plan_records:
                raise
            # Randomly sample another video
            return self[np.random.randint(len(self))]


def get_generic_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    sampler: Optional[Any] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    **kwargs,  # Ignore extra arguments
) -> DataLoader:
    """Create DataLoader with commonly used parameters.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        sampler: Optional sampler for data loading
        num_workers: Number of worker processes
        pin_memory: Pin memory for CUDA transfer
        drop_last: Drop incomplete last batch
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        collate_fn: Custom collate function
        **kwargs: Extra arguments (ignored)

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # False when using sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


class EpochAwareDistributedSampler(DistributedSampler):
    """DistributedSampler that reshuffles on every dataloader epoch.

    The imaginaire trainer recreates ``iter(dataloader_train)`` when the loader is
    exhausted but never calls ``set_epoch``, so a vanilla DistributedSampler with
    shuffle=True replays the identical permutation every epoch. torch's
    DistributedSampler builds the full index list eagerly inside ``__iter__``, so
    bumping the epoch right after delegating affects the next epoch only.
    """

    def __iter__(self):
        iterator = super().__iter__()
        self.set_epoch(self.epoch + 1)
        return iterator


def get_sampler(dataset) -> DistributedSampler:
    """Create a distributed sampler for the dataset."""
    return EpochAwareDistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def get_train_val_dataloaders(
    dataset_path: str, val_percentage: float, seed: int, video_size: tuple[int, int] = (704, 1280)
):
    video_dir = os.path.join(dataset_path, "videos")
    if not os.path.exists(video_dir):
        log.debug(f"Dataset path {dataset_path} does not exist, returning empty dataloaders")
        return dict(), dict()
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    random.seed(seed)
    random.shuffle(video_paths)

    cutoff = int(len(video_paths) * val_percentage)
    val_video_paths = video_paths[:cutoff]
    train_video_paths = video_paths[cutoff:]

    def get_dataset(video_paths):
        return L(VideoDataset)(
            video_paths=video_paths,
            num_frames=93,
            video_size=video_size,
            dataset_dir=dataset_path,
        )

    ipn_hand_train_dataset = get_dataset(train_video_paths)
    ipn_hand_val_dataset = get_dataset(val_video_paths)

    def get_dataloader(dataset):
        return L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset),
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        )

    return get_dataloader(ipn_hand_train_dataset), get_dataloader(ipn_hand_val_dataset)
