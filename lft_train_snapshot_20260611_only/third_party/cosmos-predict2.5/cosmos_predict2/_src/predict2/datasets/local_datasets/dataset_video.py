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
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from megatron.core import parallel_state
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        prompt_type: str | None = None,  # "long", "short", "medium", or None for auto
        caption_format: str = "auto",  # "text", "json", or "auto"
        video_paths: Optional[list[str]] = None,
        target_mask_dir: Optional[str] = None,
        target_mask_default_to_zero: bool = True,
        target_feature_dir: Optional[str] = None,
        target_feature_default_to_zero: bool = True,
        target_feature_dim: int = 256,
        target_feature_max_tokens: int = 64,
        target_prompt_suffix: str = "",
        target_mask_dropout_prob: float = 0.0,
        strip_tgt_token: bool = False,
        exclude_video_stems_file: Optional[str] = None,
        frame_stride: int = 1,
        frame_stride_choices: Optional[list[int]] = None,
        frame_start_policy: str = "random",
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
        self.target_mask_dir = self._resolve_target_mask_dir(target_mask_dir)
        self.target_mask_default_to_zero = target_mask_default_to_zero
        target_feature_dir_str = str(target_feature_dir) if target_feature_dir is not None else None
        self.target_feature_enabled = target_feature_dir_str is not None and target_feature_dir_str.lower() != "none"
        self.target_feature_dir = self._resolve_target_feature_dir(target_feature_dir_str)
        self.target_feature_default_to_zero = target_feature_default_to_zero
        self.target_feature_dim = int(target_feature_dim)
        self.target_feature_max_tokens = int(target_feature_max_tokens)
        self.target_prompt_suffix = target_prompt_suffix
        self.target_mask_dropout_prob = float(target_mask_dropout_prob)
        self.strip_tgt_token = strip_tgt_token
        self.exclude_video_stems_file = exclude_video_stems_file
        assert 0.0 <= self.target_mask_dropout_prob <= 1.0, "target_mask_dropout_prob must be in [0,1]"
        assert self.target_feature_dim > 0, "target_feature_dim must be positive"
        assert self.target_feature_max_tokens >= 0, "target_feature_max_tokens must be >= 0"
        # Temporal sub-sampling: 33 contiguous frames only cover ~2s of source
        # video at 15 fps, so the model never sees the full task arc and the
        # robot motion looks slow. With stride k, 33 frames span 33*k source
        # frames -- choose k to cover the manipulation task length.
        if frame_stride_choices is not None and len(frame_stride_choices) > 0:
            assert all(int(s) >= 1 for s in frame_stride_choices), "frame_stride_choices must be >=1"
            self.frame_stride_choices = [int(s) for s in frame_stride_choices]
        else:
            assert int(frame_stride) >= 1, "frame_stride must be >=1"
            self.frame_stride_choices = [int(frame_stride)]
        assert frame_start_policy in {"random", "range_start"}, "frame_start_policy must be 'random' or 'range_start'"
        self.frame_start_policy = frame_start_policy

        # Determine caption format and directory
        self._setup_caption_format()

        video_dir = os.path.join(self.dataset_dir, "videos")

        if video_paths is None:
            self.video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
            self.video_paths = sorted(self.video_paths)
        else:
            self.video_paths = video_paths
        self.video_paths = self._filter_excluded_video_stems(self.video_paths)
        log.info(f"{len(self.video_paths)} videos in total")
        self.frame_ranges = self._load_frame_ranges()

        self.num_failed_loads = 0
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])
        self.mask_size = (video_size[0], video_size[1])

    def _filter_excluded_video_stems(self, video_paths: list[str]) -> list[str]:
        """Drop split entries listed in an optional newline-delimited stem file."""
        if not self.exclude_video_stems_file:
            return video_paths
        exclude_path = self.exclude_video_stems_file
        if exclude_path.lower() == "auto":
            exclude_path = os.path.join(self.dataset_dir, "exclude_no_tgt_stems.txt")
        if not os.path.exists(exclude_path):
            log.warning(f"exclude_video_stems_file does not exist: {exclude_path}")
            return video_paths
        with open(exclude_path, "r") as f:
            excluded = {line.strip() for line in f if line.strip() and not line.startswith("#")}
        if not excluded:
            return video_paths
        filtered = [
            path for path in video_paths if os.path.splitext(os.path.basename(path))[0] not in excluded
        ]
        log.info(f"Filtered {len(video_paths) - len(filtered)} videos listed in {exclude_path}")
        return filtered

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _resolve_target_mask_dir(self, target_mask_dir: Optional[str]) -> Optional[str]:
        """Resolve optional target-mask directory for TAViD-style conditioning."""
        if target_mask_dir is None:
            for dirname in ("masks", "target_masks"):
                candidate = os.path.join(self.dataset_dir, dirname)
                if os.path.isdir(candidate):
                    return candidate
            return None
        if target_mask_dir.lower() == "none":
            return None
        if target_mask_dir.lower() == "auto":
            for dirname in ("masks", "target_masks"):
                candidate = os.path.join(self.dataset_dir, dirname)
                if os.path.isdir(candidate):
                    return candidate
            return None
        return target_mask_dir

    def _resolve_target_feature_dir(self, target_feature_dir: Optional[str]) -> Optional[str]:
        """Resolve optional target-feature directory for implicit target conditioning."""
        if target_feature_dir is None:
            return None
        if target_feature_dir.lower() == "none":
            return None
        if target_feature_dir.lower() == "auto":
            for dirname in ("target_features", "instructsam_features", "features", "target_feature"):
                candidate = os.path.join(self.dataset_dir, dirname)
                if os.path.isdir(candidate):
                    return candidate
            return None
        # A literal (non-auto) name is resolved relative to the dataset dir so a
        # bare subdir name like "target_features_multisource" works regardless of
        # the process CWD; absolute paths pass through unchanged.
        if not os.path.isabs(target_feature_dir):
            return os.path.join(self.dataset_dir, target_feature_dir)
        return target_feature_dir

    def _load_frame_ranges(self) -> dict[str, list[tuple[int, int]]]:
        """Load optional per-video frame ranges used to avoid static lead-in/tail frames."""
        ranges_path = os.path.join(self.dataset_dir, "frame_ranges.json")
        if not os.path.exists(ranges_path):
            return {}
        try:
            with open(ranges_path, "r") as f:
                raw_ranges = json.load(f)
        except Exception as exc:
            log.warning(f"Failed to read frame ranges from {ranges_path}: {exc}")
            return {}

        frame_ranges: dict[str, list[tuple[int, int]]] = {}
        for name, ranges in raw_ranges.items():
            clean_ranges = []
            for item in ranges:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                start, end = int(item[0]), int(item[1])
                if end >= start:
                    clean_ranges.append((start, end))
            if clean_ranges:
                frame_ranges[str(name)] = clean_ranges
        log.info(f"Loaded frame ranges for {len(frame_ranges)} videos from {ranges_path}")
        return frame_ranges

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float, np.ndarray, int]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        video_basename = os.path.basename(video_path).replace(".mp4", "")
        ranges = self.frame_ranges.get(video_basename, [(0, total_frames - 1)])
        ranges = [
            (max(0, start), min(total_frames - 1, end))
            for start, end in ranges
            if min(total_frames - 1, end) >= max(0, start)
        ]
        if not ranges:
            ranges = [(0, total_frames - 1)]

        # Pick a stride and range that fit; this lets curated datasets provide
        # motion-only frame ranges without physically cutting every mp4.
        range_candidates = []
        for stride_choice in self.frame_stride_choices:
            span_choice = stride_choice * (self.sequence_length - 1) + 1
            for range_start, range_end in ranges:
                if span_choice <= (range_end - range_start + 1):
                    range_candidates.append((stride_choice, range_start, range_end))

        if range_candidates:
            stride, range_start, range_end = range_candidates[np.random.randint(len(range_candidates))]
            span = stride * (self.sequence_length - 1) + 1
            if self.frame_start_policy == "range_start":
                start_frame = range_start
            else:
                max_start_idx = range_end - span + 1
                start_frame = np.random.randint(range_start, max_start_idx + 1)
        else:
            if self.frame_start_policy == "range_start":
                range_start, range_end = max(ranges, key=lambda item: item[1] - item[0])
                range_len = range_end - range_start + 1
                candidates = [
                    s for s in self.frame_stride_choices if (s * (self.sequence_length - 1) + 1) <= range_len
                ]
                if candidates:
                    stride = int(np.random.choice(candidates))
                elif range_len >= self.sequence_length:
                    stride = 1
                else:
                    candidates = [
                        s for s in self.frame_stride_choices if (s * (self.sequence_length - 1) + 1) <= total_frames
                    ]
                    stride = int(np.random.choice(candidates)) if candidates else max(1, (total_frames - 1) // max(1, (self.sequence_length - 1)))
                    range_start = 0
                start_frame = range_start
            else:
                candidates = [
                    s for s in self.frame_stride_choices if (s * (self.sequence_length - 1) + 1) <= total_frames
                ]
                if not candidates:
                    stride = max(1, (total_frames - 1) // max(1, (self.sequence_length - 1)))
                else:
                    stride = int(np.random.choice(candidates))
                span = stride * (self.sequence_length - 1) + 1
                max_start_idx = total_frames - span
                start_frame = np.random.randint(0, max_start_idx + 1)
        frame_ids = (start_frame + stride * np.arange(self.sequence_length)).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps, np.asarray(frame_ids, dtype=np.int64), total_frames

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

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float, np.ndarray, int]:
        frames, fps, frame_ids, total_frames = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps, frame_ids, total_frames

    def _target_mask_path_candidates(self, video_basename: str) -> list[str]:
        if self.target_mask_dir is None:
            return []
        return [
            os.path.join(self.target_mask_dir, f"{video_basename}{ext}")
            for ext in (".mp4", ".npz", ".png", ".jpg", ".jpeg", ".webp")
        ]

    def _target_feature_path_candidates(self, video_basename: str) -> list[str]:
        if self.target_feature_dir is None:
            return []
        return [
            os.path.join(self.target_feature_dir, f"{video_basename}{ext}")
            for ext in (".pt", ".pth", ".npy", ".npz")
        ]

    def _zero_target_feature(self) -> torch.Tensor:
        num_tokens = self.target_feature_max_tokens if self.target_feature_max_tokens > 0 else 1
        return torch.zeros(num_tokens, self.target_feature_dim, dtype=torch.float32)

    def _select_feature_from_mapping(self, feature_dict: dict[str, Any]) -> Any:
        for key in ("target_feature", "features", "feature_B_L_D", "feature", "seg_output_embeddings"):
            if key in feature_dict:
                return feature_dict[key]
        for value in feature_dict.values():
            if isinstance(value, (torch.Tensor, np.ndarray, list, tuple)):
                return value
        raise ValueError("No tensor-like feature found in target feature file")

    def _normalize_target_feature(self, feature: Any, feature_path: str) -> torch.Tensor:
        if isinstance(feature, dict):
            feature = self._select_feature_from_mapping(feature)
        if isinstance(feature, np.ndarray) and feature.dtype == object and feature.size == 1:
            feature = feature.reshape(()).item()
            if isinstance(feature, dict):
                feature = self._select_feature_from_mapping(feature)
        if isinstance(feature, torch.Tensor):
            feature_tensor = feature.detach().cpu().float()
        else:
            feature_tensor = torch.as_tensor(feature, dtype=torch.float32)

        if feature_tensor.ndim == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        elif feature_tensor.ndim == 3 and feature_tensor.shape[0] == 1:
            feature_tensor = feature_tensor[0]
        elif feature_tensor.ndim > 2:
            feature_tensor = feature_tensor.reshape(-1, feature_tensor.shape[-1])
        if feature_tensor.ndim != 2:
            raise ValueError(f"Unsupported target feature shape {tuple(feature_tensor.shape)} in {feature_path}")
        if feature_tensor.shape[-1] != self.target_feature_dim:
            raise ValueError(
                f"Target feature dim mismatch in {feature_path}: expected {self.target_feature_dim}, "
                f"got {feature_tensor.shape[-1]}"
            )

        feature_tensor = torch.nan_to_num(feature_tensor.contiguous())
        if self.target_feature_max_tokens > 0:
            num_tokens = feature_tensor.shape[0]
            if num_tokens > self.target_feature_max_tokens:
                feature_tensor = feature_tensor[: self.target_feature_max_tokens]
            elif num_tokens < self.target_feature_max_tokens:
                pad = torch.zeros(
                    self.target_feature_max_tokens - num_tokens,
                    self.target_feature_dim,
                    dtype=feature_tensor.dtype,
                )
                feature_tensor = torch.cat([feature_tensor, pad], dim=0)
        return feature_tensor

    def _load_target_feature(self, video_basename: str) -> torch.Tensor:
        """Load precomputed InstructSAM target features as [L,D]."""
        feature_path = next(
            (path for path in self._target_feature_path_candidates(video_basename) if os.path.exists(path)),
            None,
        )
        if feature_path is None:
            if self.target_feature_default_to_zero:
                return self._zero_target_feature()
            raise FileNotFoundError(
                f"Target feature for {video_basename} not found in "
                f"{self.target_feature_dir or 'auto-resolved feature directories'}"
            )

        if feature_path.endswith((".pt", ".pth")):
            raw_feature = torch.load(feature_path, map_location="cpu")
        elif feature_path.endswith(".npy"):
            raw_feature = np.load(feature_path, allow_pickle=True)
        elif feature_path.endswith(".npz"):
            with np.load(feature_path, allow_pickle=True) as feature_npz:
                raw_feature = self._select_feature_from_mapping({key: feature_npz[key] for key in feature_npz.files})
        else:
            raise ValueError(f"Unsupported target feature extension: {feature_path}")
        return self._normalize_target_feature(raw_feature, feature_path)

    def _resize_binary_mask_video(self, mask: torch.Tensor) -> torch.Tensor:
        """Resize [T,1,H,W] mask video and return [1,T,H,W]."""
        mask = torch.stack(
            [TF.resize(frame, self.mask_size, interpolation=InterpolationMode.NEAREST) for frame in mask.float()]
        )
        mask = (mask > 0.5).float()
        return mask.permute(1, 0, 2, 3).contiguous()

    def _load_npz_target_mask(
        self,
        mask_path: str,
        frame_ids: np.ndarray,
        video_frame_count: int | None = None,
    ) -> torch.Tensor:
        mask_npz = np.load(mask_path, allow_pickle=True)
        if "masks_packed" in mask_npz.files and "shape" in mask_npz.files:
            shape = tuple(int(dim) for dim in mask_npz["shape"].tolist())
            if len(shape) != 3:
                raise ValueError(f"Unsupported packed target mask shape {shape} in {mask_path}")
            flat_pixels = int(np.prod(shape[1:]))
            unpacked = np.unpackbits(mask_npz["masks_packed"], axis=1)[:, :flat_pixels]
            mask_arr = unpacked.reshape(shape)
        else:
            key = "masks" if "masks" in mask_npz.files else mask_npz.files[0]
            mask_arr = mask_npz[key]
        if mask_arr.ndim == 5:
            # RoboInter SAM files are typically [N,T,1,H,W]. Merge annotated target
            # masks when multiple instances are present.
            mask_arr = mask_arr.max(axis=0)
        if mask_arr.ndim == 4:
            if mask_arr.shape[1] == 1:  # [T,1,H,W]
                mask_arr = mask_arr[:, 0]
            elif mask_arr.shape[0] == 1:  # [1,T,H,W]
                mask_arr = mask_arr[0]
            else:
                mask_arr = mask_arr.max(axis=0)
        if mask_arr.ndim == 2:
            mask_arr = np.repeat(mask_arr[None], len(frame_ids), axis=0)
        if mask_arr.ndim != 3:
            raise ValueError(f"Unsupported target mask shape {mask_arr.shape} in {mask_path}")
        if (
            video_frame_count is not None
            and video_frame_count > 1
            and mask_arr.shape[0] > 1
            and mask_arr.shape[0] != video_frame_count
        ):
            valid_frame_ids = np.rint(frame_ids * (mask_arr.shape[0] - 1) / (video_frame_count - 1)).astype(
                np.int64
            )
        else:
            valid_frame_ids = frame_ids
        valid_frame_ids = np.clip(valid_frame_ids, 0, mask_arr.shape[0] - 1)
        mask_arr = mask_arr[valid_frame_ids]
        mask = torch.from_numpy(mask_arr).unsqueeze(1).float()  # [T,1,H,W]
        return self._resize_binary_mask_video(mask)

    def _load_target_mask(
        self,
        video_basename: str,
        frame_ids: np.ndarray,
        video_frame_count: int | None = None,
    ) -> torch.Tensor:
        """Load target mask as [1,T,H,W].

        If a mask video exists, it is sampled with the same frame ids as the RGB video.
        If a single image mask exists, it is treated like TAViD's initial-frame target mask
        and placed on the first sampled frame only; all future frames are zero.
        """
        T_frames = len(frame_ids)
        zero_mask = torch.zeros(1, T_frames, *self.mask_size, dtype=torch.float32)
        mask_path = next((path for path in self._target_mask_path_candidates(video_basename) if os.path.exists(path)), None)
        if mask_path is None:
            if self.target_mask_default_to_zero:
                return zero_mask
            raise FileNotFoundError(f"Target mask for {video_basename} not found in {self.target_mask_dir}")

        if mask_path.endswith(".mp4"):
            mask_reader = VideoReader(mask_path, ctx=cpu(0), num_threads=1)
            valid_frame_ids = np.clip(frame_ids, 0, len(mask_reader) - 1).tolist()
            mask_frames = mask_reader.get_batch(valid_frame_ids).asnumpy()
            mask_reader.seek(0)
            del mask_reader
            mask = torch.from_numpy(mask_frames[..., 0]).unsqueeze(1).float() / 255.0  # [T,1,H,W]
            return self._resize_binary_mask_video(mask)

        if mask_path.endswith(".npz"):
            return self._load_npz_target_mask(mask_path, frame_ids, video_frame_count)

        mask_img = Image.open(mask_path).convert("L")
        mask = TF.to_tensor(mask_img)
        mask = TF.resize(mask, self.mask_size, interpolation=InterpolationMode.NEAREST)
        zero_mask[:, 0] = (mask > 0.5).float()
        return zero_mask

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = dict()
            video, fps, frame_ids, total_frames = self._get_frames(self.video_paths[index])
            video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]

            # Load caption based on format
            video_path = self.video_paths[index]
            video_basename = os.path.basename(video_path).replace(".mp4", "")

            if self.caption_format == "json":
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.json")
                caption = self._load_json_caption(Path(caption_path))
            else:  # text format
                caption_path = os.path.join(self.caption_dir, f"{video_basename}.txt")
                caption = self._load_text(Path(caption_path))
            if self.strip_tgt_token:
                caption = " ".join(caption.replace("[TGT]", "").split())
            # CFG-style joint dropout: with prob `target_mask_dropout_prob`,
            # zero out the target mask AND drop the prompt suffix so the model
            # also sees the base caption distribution without mask guidance.
            drop_mask = (
                (self.target_mask_dir is not None or self.target_prompt_suffix)
                and self.target_mask_dropout_prob > 0
                and random.random() < self.target_mask_dropout_prob
            )

            if self.target_prompt_suffix and not drop_mask and "[TGT]" not in caption:
                caption = f"{caption.rstrip()} {self.target_prompt_suffix.strip()}".strip()

            data["video"] = video
            data["ai_caption"] = caption
            if self.target_mask_dir is not None or self.target_prompt_suffix:
                if drop_mask:
                    data["target_mask"] = torch.zeros(
                        1, len(frame_ids), *self.mask_size, dtype=torch.float32
                    )
                else:
                    data["target_mask"] = self._load_target_mask(video_basename, frame_ids, total_frames)
            if self.target_feature_enabled:
                data["target_feature"] = (
                    self._zero_target_feature()
                    if drop_mask
                    else self._load_target_feature(video_basename)
                )
            if not drop_mask and "[TGT]" in caption:
                data["tgt_token_text"] = "[TGT]"

            _, _, h, w = video.shape

            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception as e:
            if self.target_feature_enabled and isinstance(e, FileNotFoundError) and "Target feature" in str(e):
                raise
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]


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


def get_sampler(dataset) -> DistributedSampler:
    """Create a distributed sampler for the dataset."""
    return DistributedSampler(
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
