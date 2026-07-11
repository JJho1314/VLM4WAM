import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .lerobot.datasets.video_utils import set_frame_cache_dir
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

_SEMANTIC_PLAN_UTILS = None


def _get_work_dir():
    """Use FastWAM's registered work dir without requiring its S3 dependency."""
    try:
        from fastwam.utils import misc
    except ModuleNotFoundError as error:
        if error.name != "boto3":
            raise
        return os.environ.get("FASTWAM_WORK_DIR", "./runs")
    return misc.get_work_dir()


def _load_semantic_plan_utils():
    global _SEMANTIC_PLAN_UTILS
    if _SEMANTIC_PLAN_UTILS is not None:
        return _SEMANTIC_PLAN_UTILS

    try:
        from cosmos_predict2._src.predict2.networks.semantic_plan_conditioning import (
            load_semantic_plan_payload,
            semantic_plan_times_from_frame_indices,
        )

        _SEMANTIC_PLAN_UTILS = (load_semantic_plan_payload, semantic_plan_times_from_frame_indices)
        return _SEMANTIC_PLAN_UTILS
    except ModuleNotFoundError as import_error:
        candidates = []
        env_repo = os.environ.get("COSMOS_REPO")
        if env_repo:
            candidates.append(
                Path(env_repo) / "cosmos_predict2/_src/predict2/networks/semantic_plan_conditioning.py"
            )
        this_file = Path(__file__).resolve()
        if len(this_file.parents) > 5:
            candidates.append(
                this_file.parents[5]
                / "cosmos-predict2.5/cosmos_predict2/_src/predict2/networks/semantic_plan_conditioning.py"
            )

        for module_path in candidates:
            if not module_path.is_file():
                continue
            spec = importlib.util.spec_from_file_location(
                "fastwam_semantic_plan_conditioning", module_path
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            _SEMANTIC_PLAN_UTILS = (
                module.load_semantic_plan_payload,
                module.semantic_plan_times_from_frame_indices,
            )
            return _SEMANTIC_PLAN_UTILS
        raise import_error

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        frame_cache_dir: Optional[str] = None, # pre-decoded frame cache dir; None = decode mp4 at runtime (back-compatible)
        condition_frame_augmentation: Optional[dict] = None,
        video_augmentation: Optional[dict] = None,
        semantic_plan_dir: Optional[str] = None,
        semantic_plan_manifest: Optional[str] = None,
        semantic_plan_dim: int = 1152,
        semantic_plan_max_tokens: int = 0,
        semantic_plan_default_to_zero: bool = False,
    ):
        # Set the pre-decoded frame cache dir for this process/worker. This must run
        # before any video decode. The FASTWAM_FRAME_CACHE_DIR env var also works
        # standalone; an explicit arg here overrides it. Passing None leaves whatever
        # the env var set (so env-only usage stays back-compatible).
        if frame_cache_dir is not None:
            set_frame_cache_dir(frame_cache_dir)
        self.frame_cache_dir = frame_cache_dir

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.semantic_plan_dir = None if semantic_plan_dir is None else os.fspath(semantic_plan_dir)
        self.semantic_plan_dim = int(semantic_plan_dim)
        self.semantic_plan_max_tokens = int(semantic_plan_max_tokens)
        self.semantic_plan_default_to_zero = bool(semantic_plan_default_to_zero)
        self.semantic_plan_records = self._load_semantic_plan_manifest(semantic_plan_manifest)
        augmentation_cfg = video_augmentation if video_augmentation is not None else condition_frame_augmentation
        if augmentation_cfg is not None and is_training_set:
            if isinstance(augmentation_cfg, torch.nn.Module):
                self.video_augmentation = augmentation_cfg
            else:
                self.video_augmentation = instantiate(augmentation_cfg)
        else:
            self.video_augmentation = None

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = _get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = _get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

    def _resolve_manifest_paths(self, semantic_plan_manifest):
        if semantic_plan_manifest is None:
            return []
        if self.semantic_plan_dir is None:
            raise ValueError("semantic_plan_manifest requires semantic_plan_dir.")

        if isinstance(semantic_plan_manifest, (str, os.PathLike)):
            manifest_items = [
                item.strip()
                for item in os.fspath(semantic_plan_manifest).split(",")
                if item.strip()
            ]
        else:
            manifest_items = list(semantic_plan_manifest)

        paths = []
        for item in manifest_items:
            text = os.fspath(item)
            base = text if os.path.isabs(text) else os.path.join(self.semantic_plan_dir, text)
            matches = sorted(glob.glob(base))
            if not matches:
                raise FileNotFoundError(f"semantic plan manifest not found: {base}")
            paths.extend(matches)
        return paths

    def _load_semantic_plan_manifest(self, semantic_plan_manifest):
        paths = self._resolve_manifest_paths(semantic_plan_manifest)
        records = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "sample_id" not in record:
                        raise ValueError(f"{path}:{line_no} missing sample_id")
                    records.append(record)
        if records:
            logger.info("Loaded %d semantic-plan manifest records from %d file(s).", len(records), len(paths))
        return records

    def _semantic_record_for_index(self, idx):
        if not self.semantic_plan_records:
            return None
        return self.semantic_plan_records[int(idx)]

    @staticmethod
    def _semantic_sample_idx(record, fallback_idx):
        if record is None:
            return int(fallback_idx)
        for key in ("idx", "sample_idx", "dataset_index", "lerobot_index", "base_idx"):
            if key in record:
                return int(record[key])
        return int(fallback_idx)

    def _semantic_plan_path(self, sample_id):
        if self.semantic_plan_dir is None:
            return None
        sample_id = str(sample_id)
        root = Path(self.semantic_plan_dir)
        raw = Path(sample_id)
        candidates = [raw if raw.is_absolute() else root / raw]
        if raw.suffix not in (".pt", ".pth", ".npy", ".npz"):
            candidates.extend(root / f"{sample_id}{ext}" for ext in (".pt", ".pth", ".npy", ".npz"))
        for path in candidates:
            if path.is_file():
                return path
        return None

    def _zero_semantic_plan(self):
        if self.semantic_plan_dim <= 0:
            raise ValueError("semantic_plan_default_to_zero requires semantic_plan_dim > 0.")
        length = max(int(self.semantic_plan_max_tokens), 1)
        return torch.zeros(length, int(self.semantic_plan_dim), dtype=torch.float32)

    def _semantic_times_from_record(self, record, fallback_times):
        if record is None:
            return fallback_times
        explicit = record.get("semantic_plan_times", record.get("keyframe_times", None))
        if explicit is not None:
            return torch.as_tensor(explicit, dtype=torch.float32)
        _, semantic_plan_times_from_frame_indices = _load_semantic_plan_utils()
        record_times = semantic_plan_times_from_frame_indices(
            record.get("future_frame_indices"),
            record.get("video_frame_indices"),
        )
        return fallback_times if record_times is None else record_times

    def _attach_semantic_plan(self, data, semantic_record, sample_idx):
        if self.semantic_plan_dir is None:
            return
        load_semantic_plan_payload, _ = _load_semantic_plan_utils()
        sample_id = (
            semantic_record.get("sample_id")
            if semantic_record is not None
            else str(int(sample_idx))
        )
        plan_path = self._semantic_plan_path(sample_id)
        if plan_path is None:
            if not self.semantic_plan_default_to_zero:
                raise FileNotFoundError(f"Missing semantic plan for sample_id={sample_id}")
            semantic_plan = self._zero_semantic_plan()
            payload_times = None
        else:
            semantic_plan, payload_times = load_semantic_plan_payload(
                plan_path,
                semantic_plan_dim=self.semantic_plan_dim,
                max_tokens=self.semantic_plan_max_tokens,
            )

        data["semantic_plan"] = semantic_plan
        semantic_times = self._semantic_times_from_record(semantic_record, payload_times)
        if semantic_times is not None:
            data["semantic_plan_times"] = semantic_times
        if semantic_record is not None:
            data["semantic_plan_meta"] = dict(semantic_record)
        
    def __len__(self):
        return len(self.semantic_plan_records) if self.semantic_plan_records else len(self.lerobot_dataset)

    def _get(self, idx):
        query_idx = int(idx)
        semantic_record = self._semantic_record_for_index(query_idx)
        sample_idx = self._semantic_sample_idx(semantic_record, query_idx)
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            if self.semantic_plan_records:
                query_idx = int(np.random.randint(len(self.semantic_plan_records)))
                semantic_record = self._semantic_record_for_index(query_idx)
                sample_idx = self._semantic_sample_idx(semantic_record, query_idx)
            else:
                sample_idx = int(np.random.randint(len(self.lerobot_dataset)))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.video_augmentation is not None:
            video = self.video_augmentation(video)

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "instruction": str(task),
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        self._attach_semantic_plan(data, semantic_record, sample_idx)
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
