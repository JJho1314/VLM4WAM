# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Predict2 post-training experiments on RoboInter-Data DROID lerobot subset.

Native lerobot DROID resolution is 320x180 @ 10 fps. We crop the height to 176
(divisible by 16) and feed `num_frames=33` clips, which keeps the pixel budget
~1/6 of the GR1 480 setup so 8x A6000/H100 can fit batch-per-GPU=1 with CP=1.

Two configs are exposed:
  * `predict2_video2world_training_2b_robointer_droid_sanity`  - 5 iter, 200 ep
  * `predict2_video2world_training_2b_robointer_droid`         - longer run
"""

import os
import copy

from hydra.core.config_store import ConfigStore
from torch.utils.data import ConcatDataset

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT_2B = MODEL_CHECKPOINTS[ModelKey(post_trained=False)]

# DROID native is 320x180 @ 10 fps. 180 isn't a multiple of 16, so use 176.
_DATASET_DIR_SANITY = "/data/user/jhe724/workspace/datasets/robointer_droid_sanity"
_DATASET_DIR_FULL = "/data/user/jhe724/workspace/datasets/robointer_droid"
_DATASET_DIR_TAVID_PRIMARY = os.environ.get(
    "ROBOINTER_DROID_TAVID_PRIMARY_DIR",
    "/data/user/jhe724/workspace/datasets/robointer_droid_tavid_primary",
)

# droid_success is the 1280x720 @ 15 fps lerobot v3.0 release; we resize to
# 560x1008 (16-aligned, ~720p detail retained) and feed 33-frame clips.
_DATASET_DIR_DROID_SUCCESS = "/data/user/jhe724/workspace/datasets/droid_success_left"
_DATASET_DIR_DROID_SUCCESS_TRAIN = "/data/user/jhe724/workspace/datasets/droid_success_left_train"
_DATASET_DIR_DROID_SUCCESS_TEST = "/data/user/jhe724/workspace/datasets/droid_success_left_test"
_DATASET_DIR_DROID_SUCCESS_TRAIN_480 = "/data/user/jhe724/workspace/datasets/droid_success_left_train_480x864"
_DATASET_DIR_DROID_SUCCESS_TEST_480 = "/data/user/jhe724/workspace/datasets/droid_success_left_test_480x864"
_DATASET_DIR_DROID_SUCCESS_V21_TAVID = os.environ.get(
    "DROID_SUCCESS_V21_TAVID_DIR",
    "/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train",
)
_DATASET_DIR_DROID_SUCCESS_V21_TAVID_VAL = os.environ.get(
    "DROID_SUCCESS_V21_TAVID_VAL_DIR",
    "/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val",
)
_DATASET_DIR_DROID_FAILURE_ALL = "/data/user/jhe724/workspace/datasets/droid_failure_left_all"
_DATASET_DIR_DROID_FAILURE_CLEAN = "/data/user/jhe724/workspace/datasets/droid_failure_left_all_clean"
_DATASET_DIR_DROID_FAILURE_CLEAN_480 = "/data/user/jhe724/workspace/datasets/droid_failure_left_all_clean_480x864"
_DROID_VIDEO_SIZE_480 = (480, 864)
_DROID_SUCCESS_V21_TAVID_NUM_FRAMES = int(os.environ.get("DROID_SUCCESS_V21_TAVID_NUM_FRAMES", "49"))
_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES = [
    int(item)
    for item in os.environ.get("DROID_SUCCESS_V21_TAVID_FRAME_STRIDES", "2,3,4").split(",")
    if item.strip()
]
_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY = os.environ.get(
    "DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY",
    "range_start",
)
_DROID_SUCCESS_ITER_10000 = (
    "/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success/cosmos_predict_v2p5/"
    "video2world/2b_droid_success_560/checkpoints/iter_000010000"
)

_video_dataset_droid_sanity = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_SANITY,
    num_frames=33,
    video_size=(176, 320),
)
_dataloader_train_droid_sanity = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_sanity,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_sanity),
    batch_size=1,
    drop_last=True,
    num_workers=2,
    pin_memory=True,
)

_video_dataset_droid_full = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_FULL,
    num_frames=33,
    video_size=(176, 320),
)
_dataloader_train_droid_full = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_full,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_full),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

_video_dataset_droid_full_tavid_mask = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_TAVID_PRIMARY,
    num_frames=33,
    video_size=(176, 320),
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
)
_dataloader_train_droid_full_tavid_mask = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_full_tavid_mask,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_full_tavid_mask),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

# v2: target-mask metadata + temporal sub-sampling so the
# 33-frame training clip spans the whole DROID task arc rather than ~2 s of
# slow motion. The mask is kept for attention supervision, not as an input
# channel, and the model learns "fast" task dynamics.
_video_dataset_droid_full_tavid_mask_v2 = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_TAVID_PRIMARY,
    num_frames=49,            # TAViD default clip length; latent_t = 13 <= base state_t=24
    video_size=(176, 320),
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    target_mask_dropout_prob=0.0,
    # DROID episodes are 200~480 frames @ 15 fps. With stride in {1, 2, 4} a
    # 49-frame clip spans 49~193 source frames (~3~13 s), multi-scale so the
    # model learns different task tempos. Stride 6 dropped (span 289 frames
    # exceeds many DROID episodes).
    frame_stride_choices=[1, 2, 4],
)
_dataloader_train_droid_full_tavid_mask_v2 = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_full_tavid_mask_v2,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_full_tavid_mask_v2),
    batch_size=1,
    drop_last=True,
    num_workers=12,           # 8 GPU * 12 workers = 96 worker processes within --cpus-per-task=96
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)


# Sanity run: load post-trained 2B, take 5 steps to verify env + data pipeline.
predict2_video2world_training_2b_robointer_droid_sanity = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_sanity,
    checkpoint=dict(
        save_iter=5,  # save once at end so we exercise the save path
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_robointer_droid_sanity",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[100],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=1,
        max_iter=5,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=2, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=10, save_s3=False),
            every_n_sample_ema=dict(every_n=10, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# Real run config (override max_iter / dataloader path via CLI when ready).
predict2_video2world_training_2b_robointer_droid = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_full,
    checkpoint=dict(
        save_iter=500,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_robointer_droid",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=10000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=500, save_s3=False),
            every_n_sample_ema=dict(every_n=500, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# TAViD-style target-mask conditioning on RoboInter/LeRobot primary videos.
# This uses RoboInter's own primary camera videos and SAM masks, which share the
# same episode ids and frame counts.
predict2_video2world_training_2b_robointer_droid_tavid_mask = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_full_tavid_mask,
    checkpoint=dict(
        save_iter=1000,
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
        dcp_allow_mismatched_size=True,
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_robointer_droid_tavid_mask_primary",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=10000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=500, save_s3=False),
            every_n_sample_ema=dict(every_n=500, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
    model=dict(
        config=dict(
            target_mask_condition_frames_only=True,
            target_attention_loss_weight=0.05,
            net=dict(
                concat_target_mask=False,
                tavid_attn_alignment_blocks=[8, 12, 16, 20],
                tavid_attn_query_chunk_size=1024,
            ),
        ),
    ),
)


_video_dataset_droid_success = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS,
    num_frames=33,
    video_size=(560, 1008),  # ~720p, 16-aligned
)
_dataloader_train_droid_success = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

_video_dataset_droid_success_train = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_TRAIN_480,
    num_frames=33,
    video_size=_DROID_VIDEO_SIZE_480,
)
_video_dataset_droid_success_test = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_TEST_480,
    num_frames=33,
    video_size=_DROID_VIDEO_SIZE_480,
)
_video_dataset_droid_failure_all = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_FAILURE_CLEAN_480,
    num_frames=33,
    video_size=_DROID_VIDEO_SIZE_480,
)
_video_dataset_droid_success_train_tavid_mask = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_TRAIN_480,
    num_frames=33,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
)
_video_dataset_droid_success_test_tavid_mask = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_TEST_480,
    num_frames=33,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
)
_video_dataset_droid_failure_all_tavid_mask = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_FAILURE_CLEAN_480,
    num_frames=33,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
)
_video_dataset_droid_success_failure = L(ConcatDataset)(
    datasets=[
        _video_dataset_droid_success_train,
        _video_dataset_droid_failure_all,
    ],
)
_video_dataset_droid_success_failure_tavid_mask = L(ConcatDataset)(
    datasets=[
        _video_dataset_droid_success_train_tavid_mask,
        _video_dataset_droid_failure_all_tavid_mask,
    ],
)
_dataloader_train_droid_success_failure = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_failure,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_failure),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
_dataloader_val_droid_success = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_test,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_test),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
_dataloader_train_droid_success_failure_tavid_mask = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_failure_tavid_mask,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_failure_tavid_mask),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
_dataloader_val_droid_success_tavid_mask = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_test_tavid_mask,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_test_tavid_mask),
    batch_size=1,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

_video_dataset_droid_success_v21_tavid_mask = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_val_tavid_mask = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID_VAL,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_instructsam_feature = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="auto",
    target_feature_default_to_zero=False,
    target_feature_dim=256,
    target_feature_max_tokens=64,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_val_instructsam_feature = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID_VAL,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="auto",
    target_feature_default_to_zero=False,
    target_feature_dim=256,
    target_feature_max_tokens=64,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
# Text-free multi-source: fused mask+detect+vtext InstructSAM feature (one [L,256]
# tensor) loaded from target_features_multisource. Budgets 16/16/32 -> 64 tokens.
_DROID_SUCCESS_V21_MULTISOURCE_SEGMENTS = [16, 16, 32]
_DROID_SUCCESS_V21_MULTISOURCE_MAX_TOKENS = sum(_DROID_SUCCESS_V21_MULTISOURCE_SEGMENTS)
_video_dataset_droid_success_v21_instructsam_multisource = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="target_features_multisource",
    target_feature_default_to_zero=False,
    target_feature_dim=256,
    target_feature_max_tokens=_DROID_SUCCESS_V21_MULTISOURCE_MAX_TOKENS,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_val_instructsam_multisource = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID_VAL,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="target_features_multisource",
    target_feature_default_to_zero=False,
    target_feature_dim=256,
    target_feature_max_tokens=_DROID_SUCCESS_V21_MULTISOURCE_MAX_TOKENS,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_baseline = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="none",
    strip_tgt_token=True,
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_val_baseline = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID_VAL,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="none",
    strip_tgt_token=True,
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_dataloader_train_droid_success_v21_tavid_mask = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_tavid_mask,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_tavid_mask),
    batch_size=1,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
_dataloader_val_droid_success_v21_tavid_mask = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_val_tavid_mask,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_val_tavid_mask),
    batch_size=1,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
_dataloader_train_droid_success_v21_instructsam_feature = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_instructsam_feature,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_instructsam_feature),
    batch_size=1,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
_dataloader_val_droid_success_v21_instructsam_feature = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_val_instructsam_feature,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_val_instructsam_feature),
    batch_size=1,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
_dataloader_train_droid_success_v21_instructsam_multisource = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_instructsam_multisource,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_instructsam_multisource),
    batch_size=1,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
_dataloader_val_droid_success_v21_instructsam_multisource = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_val_instructsam_multisource,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_val_instructsam_multisource),
    batch_size=1,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
# Dense feature-map tokens: SAM dense vision grid features selected by GT target
# mask and projected to [64,256]. The model consumes them as an 8x8 feature map
# instead of mean-pooling to one target vector.
_video_dataset_droid_success_v21_gt_mask_spatial_feature = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="target_features_gt_mask_spatial64",
    target_feature_default_to_zero=False,
    target_feature_dim=256,
    target_feature_max_tokens=64,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_val_gt_mask_spatial_feature = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID_VAL,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="target_features_gt_mask_spatial64",
    target_feature_default_to_zero=False,
    target_feature_dim=256,
    target_feature_max_tokens=64,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_dataloader_train_droid_success_v21_gt_mask_spatial_feature = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_gt_mask_spatial_feature,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_gt_mask_spatial_feature),
    batch_size=1,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
_dataloader_val_droid_success_v21_gt_mask_spatial_feature = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_val_gt_mask_spatial_feature,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_val_gt_mask_spatial_feature),
    batch_size=1,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
# Hybrid: preserve the old successful mask_query target_feature path, and add a
# second dense spatial feature map from the fine-tuned InstructSAM stage2 LoRA.
_video_dataset_droid_success_v21_hybrid_stage2_lora_feature = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="target_features",
    target_feature_default_to_zero=False,
    target_feature_dim=256,
    target_feature_max_tokens=64,
    target_dense_feature_dir="target_features_gt_mask_spatial64_instructsam_stage2_lora",
    target_dense_feature_default_to_zero=False,
    target_dense_feature_dim=256,
    target_dense_feature_max_tokens=64,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_val_hybrid_stage2_lora_feature = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID_VAL,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="target_features",
    target_feature_default_to_zero=False,
    target_feature_dim=256,
    target_feature_max_tokens=64,
    target_dense_feature_dir="target_features_gt_mask_spatial64_instructsam_stage2_lora",
    target_dense_feature_default_to_zero=False,
    target_dense_feature_dim=256,
    target_dense_feature_max_tokens=64,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_dataloader_train_droid_success_v21_hybrid_stage2_lora_feature = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_hybrid_stage2_lora_feature,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_hybrid_stage2_lora_feature),
    batch_size=1,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
_dataloader_val_droid_success_v21_hybrid_stage2_lora_feature = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_val_hybrid_stage2_lora_feature,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_val_hybrid_stage2_lora_feature),
    batch_size=1,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
# Mask-free dense InstructSAM decoder features. These are text/query-conditioned
# mask-decoder continuous maps exported as full spatial grids; Cosmos receives
# only target_feature, not target_mask.
_video_dataset_droid_success_v21_decoder_dense_maskfree_feature = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="none",
    target_feature_dir="target_features_instructsam_decoder_dense_stage2_lora",
    target_feature_default_to_zero=False,
    target_feature_dim=256,
    target_feature_max_tokens=0,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_val_decoder_dense_maskfree_feature = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID_VAL,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="none",
    target_feature_dir="target_features_instructsam_decoder_dense_stage2_lora",
    target_feature_default_to_zero=False,
    target_feature_dim=256,
    target_feature_max_tokens=0,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_dataloader_train_droid_success_v21_decoder_dense_maskfree_feature = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_decoder_dense_maskfree_feature,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_decoder_dense_maskfree_feature),
    batch_size=1,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
_dataloader_val_droid_success_v21_decoder_dense_maskfree_feature = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_val_decoder_dense_maskfree_feature,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_val_decoder_dense_maskfree_feature),
    batch_size=1,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
# raw_seg ablation data: same videos/masks, but the target feature is the RAW
# 2048-d Qwen3VL hidden state at [SEG] positions (feature_mode=raw_seg), not the
# 256-d mask_hidden_fcs projection. Precomputed into target_features_rawseg/.
_video_dataset_droid_success_v21_rawseg_feature = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="target_features_rawseg",
    target_feature_default_to_zero=False,
    target_feature_dim=2048,
    target_feature_max_tokens=64,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_val_rawseg_feature = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID_VAL,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="target_features_rawseg",
    target_feature_default_to_zero=False,
    target_feature_dim=2048,
    target_feature_max_tokens=64,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_dataloader_train_droid_success_v21_rawseg_feature = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_rawseg_feature,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_rawseg_feature),
    batch_size=1,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
_dataloader_val_droid_success_v21_rawseg_feature = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_val_rawseg_feature,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_val_rawseg_feature),
    batch_size=1,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
# Match-ground data: GT masks (matching supervision only) + raw [SEG] 2048 as
# `target_feature` (grounding WHAT) + [SEG] projection 256 as
# `target_dense_feature` (matching query WHERE). Both FT-extractor features.
_video_dataset_droid_success_v21_match_ground = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="target_features_rawseg_ft",
    target_feature_default_to_zero=False,
    target_feature_dim=2048,
    target_feature_max_tokens=16,
    target_dense_feature_dir="target_features_ft",
    target_dense_feature_default_to_zero=False,
    target_dense_feature_dim=256,
    target_dense_feature_max_tokens=16,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_video_dataset_droid_success_v21_val_match_ground = L(VideoDataset)(
    dataset_dir=_DATASET_DIR_DROID_SUCCESS_V21_TAVID_VAL,
    num_frames=_DROID_SUCCESS_V21_TAVID_NUM_FRAMES,
    video_size=_DROID_VIDEO_SIZE_480,
    target_mask_dir="auto",
    target_mask_default_to_zero=False,
    target_feature_dir="target_features_rawseg_ft",
    target_feature_default_to_zero=False,
    target_feature_dim=2048,
    target_feature_max_tokens=16,
    target_dense_feature_dir="target_features_ft",
    target_dense_feature_default_to_zero=False,
    target_dense_feature_dim=256,
    target_dense_feature_max_tokens=16,
    target_prompt_suffix="The robot interacts with the [TGT] target object.",
    exclude_video_stems_file="auto",
    frame_stride_choices=_DROID_SUCCESS_V21_TAVID_FRAME_STRIDES,
    frame_start_policy=_DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY,
)
_dataloader_train_droid_success_v21_match_ground = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_match_ground,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_match_ground),
    batch_size=1,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
_dataloader_val_droid_success_v21_match_ground = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_val_match_ground,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_val_match_ground),
    batch_size=1,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
_dataloader_train_droid_success_v21_baseline = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_baseline,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_baseline),
    batch_size=1,
    drop_last=True,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
_dataloader_val_droid_success_v21_baseline = L(get_generic_dataloader)(
    dataset=_video_dataset_droid_success_v21_val_baseline,
    sampler=L(get_sampler)(dataset=_video_dataset_droid_success_v21_val_baseline),
    batch_size=1,
    drop_last=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)


# Real run on droid_success high-res lerobot v3 dataset.
predict2_video2world_training_2b_droid_success = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_success,
    checkpoint=dict(
        save_iter=500,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_droid_success_560",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=10000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=500, save_s3=False),
            every_n_sample_ema=dict(every_n=500, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# Phase 2: resume from 10k checkpoint, enable grad_accum=4 (effective global
# batch 32), train 20k more steps to max_iter=30000.
predict2_video2world_training_2b_droid_success_phase2 = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_success,
    checkpoint=dict(
        save_iter=1000,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        # Distinct name → fresh output dir → starts from base checkpoint, not from phase 1 ckpt.
        name="2b_droid_success_560_accum4",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=20000,
        grad_accum_iter=4,  # micro-batch 1 × 8 GPU × 4 accum = effective global batch 32
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=1000, save_s3=False),
            every_n_sample_ema=dict(every_n=1000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# Post-train from the base Cosmos 2B checkpoint on droid_success train split
# plus all droid_failure, and report held-out droid_success validation loss.
predict2_video2world_training_2b_droid_success_failure = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_success_failure,
    dataloader_val=_dataloader_val_droid_success,
    checkpoint=dict(
        save_iter=1000,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_droid_success_failure_base_30k_480_clean_val1000_scratch",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=10000,
        validation_iter=10000,
        run_validation=True,
        run_validation_on_start=False,
        max_val_iter=64,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=1000, save_s3=False),
            every_n_sample_ema=dict(every_n=1000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
)


# TAViD-style target-mask supervision. Explicit target-mask input channels are
# disabled; masks are kept as metadata for cross-attention alignment.
predict2_video2world_training_2b_droid_success_failure_tavid_mask = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_success_failure_tavid_mask,
    dataloader_val=_dataloader_val_droid_success_tavid_mask,
    checkpoint=dict(
        save_iter=1000,
        # pyrefly: ignore  # missing-attribute
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
        dcp_allow_mismatched_size=True,
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_droid_success_failure_tavid_mask_480",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=10000,
        validation_iter=10000,
        run_validation=True,
        run_validation_on_start=False,
        max_val_iter=64,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=1000, save_s3=False),
            every_n_sample_ema=dict(every_n=1000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
    model=dict(
        config=dict(
            target_mask_condition_frames_only=True,
            target_attention_loss_weight=0.05,
            net=dict(
                concat_target_mask=False,
                tavid_attn_alignment_blocks=[8, 12, 16, 20],
                tavid_attn_query_chunk_size=1024,
            ),
        ),
    ),
)

predict2_video2world_training_2b_droid_success_v21_tavid_mask = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_success_v21_tavid_mask,
    dataloader_val=_dataloader_val_droid_success_v21_tavid_mask,
    checkpoint=dict(
        save_iter=1000,
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
        dcp_allow_mismatched_size=True,
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_droid_success_v21_tavid_mask_480",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=14000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=1000, save_s3=False),
            every_n_sample_ema=dict(every_n=1000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
    model=dict(
        config=dict(
            target_mask_condition_frames_only=True,
            target_attention_loss_weight=0.05,
            net=dict(
                concat_target_mask=False,
                tavid_attn_alignment_blocks=[8, 12, 16, 20],
                tavid_attn_query_chunk_size=1024,
            ),
        ),
    ),
)

predict2_video2world_training_2b_droid_success_v21_instructsam_implicit_mask = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_tavid_mask
)
predict2_video2world_training_2b_droid_success_v21_instructsam_implicit_mask["dataloader_train"] = (
    _dataloader_train_droid_success_v21_instructsam_feature
)
predict2_video2world_training_2b_droid_success_v21_instructsam_implicit_mask["dataloader_val"] = (
    _dataloader_val_droid_success_v21_instructsam_feature
)
predict2_video2world_training_2b_droid_success_v21_instructsam_implicit_mask["job"]["name"] = (
    "2b_droid_success_v21_instructsam_feature_context_480_lr_split_val1k_49f"
)
predict2_video2world_training_2b_droid_success_v21_instructsam_implicit_mask["model"]["config"]["net"].update(
    dict(
        concat_target_mask=False,
        target_mask_context_tokens=False,
        target_feature_context_tokens=True,
        target_feature_context_in_dim=256,
        target_feature_context_hidden_dim=512,
        target_feature_context_max_tokens=64,
        tavid_attn_alignment_blocks=[8, 12, 16, 20],
        tavid_attn_alignment_token_source="text_feature",
        tavid_attn_query_chunk_size=1024,
    )
)
predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context = (
    predict2_video2world_training_2b_droid_success_v21_instructsam_implicit_mask
)

predict2_video2world_training_2b_droid_success_v21_instructsam_feature_target_branch = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context
)
predict2_video2world_training_2b_droid_success_v21_instructsam_feature_target_branch["job"]["name"] = (
    "2b_droid_success_v21_instructsam_feature_target_branch_480_lr_split_val1k_49f"
)
predict2_video2world_training_2b_droid_success_v21_instructsam_feature_target_branch["model"]["config"]["net"].update(
    dict(
        target_feature_context_append_to_text=False,
        target_feature_cross_attention=True,
        target_feature_cross_attention_init_gate=0.0,
        tavid_attn_alignment_blocks=[8, 12, 16, 20],
        tavid_attn_alignment_token_source="feature",
        tavid_attn_query_chunk_size=1024,
    )
)
predict2_video2world_training_2b_droid_success_v21_instructsam_feature_target_branch["model"]["config"].update(
    dict(
        target_feature_contrastive_loss_weight=0.0,
        target_feature_contrastive_temperature=0.07,
        target_feature_contrastive_margin=0.2,
        target_feature_contrastive_margin_loss_weight=0.5,
    )
)

predict2_video2world_training_2b_droid_success_v21_dense_spatial_target = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context
)
predict2_video2world_training_2b_droid_success_v21_dense_spatial_target["job"]["name"] = (
    "2b_droid_success_v21_dense_spatial_target_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_dense_spatial_target["model"]["config"]["net"].update(
    dict(
        concat_target_mask=False,
        target_mask_context_tokens=False,
        target_feature_context_tokens=False,
        target_feature_context_append_to_text=False,
        target_feature_cross_attention=False,
        target_dense_spatial_tokens=True,
        target_dense_spatial_feature_dim=256,
        target_dense_spatial_hidden_dim=512,
        target_dense_spatial_init_gate=0.01,
        tavid_attn_alignment_blocks=[8, 12, 16, 20],
        tavid_attn_alignment_token_source="text",
        tavid_attn_query_chunk_size=1024,
    )
)

# Ablation: identical dense spatial injection, but the painted WHAT vector is the
# RAW 2048-d Qwen3VL [SEG] hidden state (full task semantics) instead of the
# 256-d mask_hidden_fcs projection (segmentation-specialized). Same extractor
# (original InstructSAM-2B), same budget — the only variable is the feature type.
predict2_video2world_training_2b_droid_success_v21_dense_spatial_rawseg = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_dense_spatial_target
)
predict2_video2world_training_2b_droid_success_v21_dense_spatial_rawseg["dataloader_train"] = (
    _dataloader_train_droid_success_v21_rawseg_feature
)
predict2_video2world_training_2b_droid_success_v21_dense_spatial_rawseg["dataloader_val"] = (
    _dataloader_val_droid_success_v21_rawseg_feature
)
predict2_video2world_training_2b_droid_success_v21_dense_spatial_rawseg["job"]["name"] = (
    "2b_droid_success_v21_dense_spatial_rawseg2048_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_dense_spatial_rawseg["model"]["config"]["net"].update(
    dict(
        target_dense_spatial_feature_dim=2048,
    )
)

# Mask-free match-ground: WHERE is learned (matching head over the [SEG]
# projection query, GT-mask supervised on ALL frames -> temporal tracking),
# WHAT is the raw [SEG] hidden read via gated attention. NO mask at inference.
predict2_video2world_training_2b_droid_success_v21_match_ground = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_tavid_mask
)
predict2_video2world_training_2b_droid_success_v21_match_ground["dataloader_train"] = (
    _dataloader_train_droid_success_v21_match_ground
)
predict2_video2world_training_2b_droid_success_v21_match_ground["dataloader_val"] = (
    _dataloader_val_droid_success_v21_match_ground
)
predict2_video2world_training_2b_droid_success_v21_match_ground["job"]["name"] = (
    "2b_droid_success_v21_match_ground_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_match_ground["model"]["config"].update(
    dict(
        # Full tracked masks as supervision (mask is NOT an input anywhere in
        # this experiment, so no train/inference mismatch).
        target_mask_condition_frames_only=False,
        target_matching_loss_weight=1.0,
    )
)
predict2_video2world_training_2b_droid_success_v21_match_ground["model"]["config"]["net"].update(
    dict(
        concat_target_mask=False,
        target_mask_concat_input=False,
        target_mask_context_tokens=False,
        target_feature_context_tokens=False,
        target_dense_spatial_tokens=False,
        target_match_ground=True,
        target_match_ground_query_dim=256,
        target_match_ground_dim=2048,
        target_match_ground_match_dim=256,
        target_match_ground_num_heads=8,
        target_match_ground_gate_init=0.0,
        target_match_ground_dropout=0.1,
        tavid_attn_alignment_blocks=[8, 12, 16, 20],
        tavid_attn_alignment_token_source="text",
        tavid_attn_query_chunk_size=1024,
    )
)

# Match-ground v2: the ONLY change from v1 is a NONLINEAR (MLP) WHERE head.
# Probes A/B/C established v1's failure was a LINEAR match_q/match_k that could
# only read the features' dominant common-mode (-> query-independent foreground
# predictor); an MLP recovers query-specific localization (+0.125 AUC gap), and
# block-0 (the current match point) is the best depth. Centering / contrastive /
# noise-weighting / depth-change were each tested and add ~0, so they are NOT
# used. Captions untouched; inference stays fully mask-free.
predict2_video2world_training_2b_droid_success_v21_match_ground_v2 = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_match_ground
)
predict2_video2world_training_2b_droid_success_v21_match_ground_v2["job"]["name"] = (
    "2b_droid_success_v21_match_ground_v2_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_match_ground_v2["model"]["config"]["net"].update(
    dict(
        target_match_ground_mlp_hidden=512,
    )
)

# Match-ground v3: v2 (MLP WHERE head) + GT-MASK GATE CURRICULUM to break the gate
# deadlock. v2 trained cleanly but the gate stayed ~0 (tanh 1e-4): the gate's
# gradient is diffusion-loss-only, and early on the predicted soft mask is too
# weak for the injection to help -> gate never opens. The curriculum gates the
# WHAT injection with the GT mask early (the dense recipe that provably drove the
# gate 0.01->1.0), then anneals to the predicted soft mask: gt_blend=1 until
# hold_iters (500), linear to 0 by gate_iters (1200), pure-predicted after. The
# matching BCE always supervises the PREDICTED logits, so the predicted mask
# keeps improving to take over the handoff. Inference uses gt_blend=0 + no GT
# mask -> still fully mask-free. (~10 epochs = 1470 iters.)
predict2_video2world_training_2b_droid_success_v21_match_ground_v3 = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_match_ground_v2
)
predict2_video2world_training_2b_droid_success_v21_match_ground_v3["job"]["name"] = (
    "2b_droid_success_v21_match_ground_v3_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_match_ground_v3["model"]["config"].update(
    dict(
        target_match_ground_gt_gate_hold_iters=500,
        target_match_ground_gt_gate_iters=1200,
    )
)

predict2_video2world_training_2b_droid_success_v21_dense_feature_map_target = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_dense_spatial_target
)
predict2_video2world_training_2b_droid_success_v21_dense_feature_map_target["dataloader_train"] = (
    _dataloader_train_droid_success_v21_gt_mask_spatial_feature
)
predict2_video2world_training_2b_droid_success_v21_dense_feature_map_target["dataloader_val"] = (
    _dataloader_val_droid_success_v21_gt_mask_spatial_feature
)
predict2_video2world_training_2b_droid_success_v21_dense_feature_map_target["job"]["name"] = (
    "2b_droid_success_v21_dense_feature_map_target_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_dense_feature_map_target["model"]["config"]["net"].update(
    dict(
        target_dense_spatial_use_feature_map=True,
    )
)

predict2_video2world_training_2b_droid_success_v21_maskfree_decoder_dense_target = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_dense_spatial_target
)
predict2_video2world_training_2b_droid_success_v21_maskfree_decoder_dense_target["dataloader_train"] = (
    _dataloader_train_droid_success_v21_decoder_dense_maskfree_feature
)
predict2_video2world_training_2b_droid_success_v21_maskfree_decoder_dense_target["dataloader_val"] = (
    _dataloader_val_droid_success_v21_decoder_dense_maskfree_feature
)
predict2_video2world_training_2b_droid_success_v21_maskfree_decoder_dense_target["job"]["name"] = (
    "2b_droid_success_v21_maskfree_decoder_dense_target_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_maskfree_decoder_dense_target["model"]["config"]["net"].update(
    dict(
        concat_target_mask=False,
        target_mask_context_tokens=False,
        target_feature_context_tokens=False,
        target_feature_context_append_to_text=False,
        target_feature_cross_attention=False,
        target_dense_spatial_tokens=True,
        target_dense_spatial_feature_dim=256,
        target_dense_spatial_hidden_dim=512,
        target_dense_spatial_init_gate=0.01,
        target_dense_spatial_use_feature_map=True,
        target_dense_spatial_mask_free=True,
        tavid_attn_alignment_blocks=[],
        tavid_attn_query_chunk_size=1024,
    )
)

predict2_video2world_training_2b_droid_success_v21_feature_input_channel_target = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_maskfree_decoder_dense_target
)
predict2_video2world_training_2b_droid_success_v21_feature_input_channel_target["job"]["name"] = (
    "2b_droid_success_v21_feature_input_channel_target_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_feature_input_channel_target["model"]["config"]["net"].update(
    dict(
        concat_target_mask=False,
        target_mask_concat_input=False,
        target_mask_context_tokens=False,
        target_feature_context_tokens=False,
        target_feature_context_append_to_text=False,
        target_feature_cross_attention=False,
        target_dense_spatial_tokens=False,
        target_dense_spatial_mask_free=False,
        target_feature_concat_input=True,
        target_feature_concat_input_dim=256,
        target_feature_concat_input_hidden_dim=128,
        target_feature_concat_input_channels=8,
        tavid_attn_alignment_blocks=[],
        tavid_attn_query_chunk_size=1024,
    )
)

predict2_video2world_training_2b_droid_success_v21_feature_control_map_channel_target = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_feature_input_channel_target
)
predict2_video2world_training_2b_droid_success_v21_feature_control_map_channel_target["job"]["name"] = (
    "2b_droid_success_v21_feature_control_map_channel_target_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_feature_control_map_channel_target["model"]["config"]["net"].update(
    dict(
        target_feature_concat_input=True,
        target_feature_concat_input_dim=256,
        target_feature_concat_input_hidden_dim=64,
        target_feature_concat_input_channels=1,
        target_feature_concat_input_mode="control_map",
        target_feature_concat_input_output_scale=1.0,
    )
)

# Mask-free latent-space grounding: InstructSAM decoder dense hidden features are
# projected into a Cosmos latent target field and injected only inside selected
# DiT blocks through gated residual modulation. No target mask logits or binary
# mask are consumed by this path at inference.
predict2_video2world_training_2b_droid_success_v21_latent_grounding_decoder_dense_target = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_maskfree_decoder_dense_target
)
predict2_video2world_training_2b_droid_success_v21_latent_grounding_decoder_dense_target["job"]["name"] = (
    "2b_droid_success_v21_latent_grounding_decoder_dense_target_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_latent_grounding_decoder_dense_target["model"]["config"]["net"].update(
    dict(
        concat_target_mask=False,
        target_mask_concat_input=False,
        target_mask_context_tokens=False,
        target_feature_context_tokens=False,
        target_feature_context_append_to_text=False,
        target_feature_cross_attention=False,
        target_feature_concat_input=False,
        target_dense_spatial_tokens=False,
        target_dense_spatial_mask_free=False,
        target_latent_grounding=True,
        target_latent_grounding_feature_dim=256,
        target_latent_grounding_hidden_dim=512,
        target_latent_grounding_blocks=[8, 12, 16, 20],
        target_latent_grounding_init_gate=0.01,
        tavid_attn_alignment_blocks=[],
        tavid_attn_query_chunk_size=1024,
    )
)

# TAViD-style target-aware, WITHOUT the explicit mask-in-latent channel: text +
# [TGT] conditioning and the TAVID attention loss as in tavid_mask, plus the mask
# entering generation as SPATIAL context tokens (TargetMaskContextAdapter:
# patchified mask + coords appended after text in cross-attention). The latent
# input is untouched (concat_target_mask=False). At inference the mask can come
# from InstructSAM (target_mask_path / target_query).
predict2_video2world_training_2b_droid_success_v21_tavid_mask_context = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_tavid_mask
)
predict2_video2world_training_2b_droid_success_v21_tavid_mask_context["job"]["name"] = (
    "2b_droid_success_v21_tavid_mask_context_480_49f"
)
predict2_video2world_training_2b_droid_success_v21_tavid_mask_context["model"]["config"]["net"].update(
    dict(
        concat_target_mask=False,             # mask stays OUT of the latent input
        target_mask_concat_input=False,
        target_mask_context_tokens=True,      # mask -> spatial context tokens
        target_mask_context_patch_size=(1, 2, 2),
        target_mask_context_hidden_dim=512,
        target_mask_context_max_tokens=256,
        target_mask_context_include_coords=True,
        tavid_attn_alignment_blocks=[8, 12, 16, 20],
        tavid_attn_alignment_token_source="text",
        tavid_attn_query_chunk_size=1024,
    )
)

# Text-free multi-source: drop the T5 text stream and condition Cosmos ONLY on the
# fused InstructSAM mask+detect+vtext feature (replace_text=True). A learned
# per-source segment embedding (segments=[16,16,32]) marks each representation.
predict2_video2world_training_2b_droid_success_v21_instructsam_textfree_multisource = copy.deepcopy(
    predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context
)
predict2_video2world_training_2b_droid_success_v21_instructsam_textfree_multisource["dataloader_train"] = (
    _dataloader_train_droid_success_v21_instructsam_multisource
)
predict2_video2world_training_2b_droid_success_v21_instructsam_textfree_multisource["dataloader_val"] = (
    _dataloader_val_droid_success_v21_instructsam_multisource
)
predict2_video2world_training_2b_droid_success_v21_instructsam_textfree_multisource["job"]["name"] = (
    "2b_droid_success_v21_instructsam_textfree_multisource_480_lr_split_val1k_49f"
)
predict2_video2world_training_2b_droid_success_v21_instructsam_textfree_multisource["model"]["config"]["net"].update(
    dict(
        concat_target_mask=False,
        target_mask_context_tokens=False,
        target_feature_context_tokens=True,
        target_feature_context_in_dim=256,
        target_feature_context_hidden_dim=512,
        target_feature_context_max_tokens=_DROID_SUCCESS_V21_MULTISOURCE_MAX_TOKENS,
        target_feature_context_replace_text=True,
        target_feature_context_append_to_text=False,
        target_feature_context_source_segments=_DROID_SUCCESS_V21_MULTISOURCE_SEGMENTS,
        # Text is gone, so the target-attention loss must align FEATURE tokens.
        tavid_attn_alignment_blocks=[8, 12, 16, 20],
        tavid_attn_alignment_token_source="feature",
        tavid_attn_query_chunk_size=1024,
    )
)

predict2_video2world_training_2b_droid_success_v21_baseline_nomask_noloss = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_success_v21_baseline,
    dataloader_val=_dataloader_val_droid_success_v21_baseline,
    checkpoint=dict(
        save_iter=1000,
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_droid_success_v21_baseline_nomask_noloss_480",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-14.5), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[1_000],
        cycle_lengths=[100000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=14000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=1000, save_s3=False),
            every_n_sample_ema=dict(every_n=1000, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
    model=dict(
        config=dict(
            target_attention_loss_weight=0.0,
            net=dict(
                tavid_attn_alignment_blocks=[],
            ),
        ),
    ),
)


# v2: full finetune on the same RoboInter DROID primary data, but with
# CFG-style mask + caption dropout, weaker (single-layer, 0.005) attention
# alignment, lower LR and fewer steps so the base autoregressive long-video
# capability is preserved while learning mask-guided manipulation.
predict2_video2world_training_2b_robointer_droid_tavid_v2 = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT_2B.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    dataloader_train=_dataloader_train_droid_full_tavid_mask_v2,
    checkpoint=dict(
        save_iter=1000,
        load_path=get_checkpoint_path(DEFAULT_CHECKPOINT_2B.s3.uri),
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
        dcp_allow_mismatched_size=True,
    ),
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_robointer_droid_tavid_v2",
        wandb_mode="online",
    ),
    optimizer=dict(lr=2 ** (-16), weight_decay=0.001),
    scheduler=dict(
        f_max=[0.5],
        f_min=[0.2],
        warm_up_steps=[500],
        cycle_lengths=[30000],
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=5000,
        straggler_detection=dict(enabled=False),
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=100, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=500, save_s3=False),
            every_n_sample_ema=dict(every_n=500, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(context_parallel_size=1),
    model=dict(
        config=dict(
            # Explicit mask-channel conditioning is disabled globally. This
            # config keeps the mask loader path off the input channel and also
            # disables attention alignment for a no-target-loss ablation.
            target_mask_condition_frames_only=True,
            target_attention_loss_weight=0.0,
            net=dict(
                concat_target_mask=False,
                tavid_attn_alignment_blocks=[],
                tavid_attn_query_chunk_size=1024,
            ),
        ),
    ),
)


cs = ConfigStore.instance()
for _item in [
    predict2_video2world_training_2b_robointer_droid_sanity,
    predict2_video2world_training_2b_robointer_droid,
    predict2_video2world_training_2b_robointer_droid_tavid_mask,
    predict2_video2world_training_2b_robointer_droid_tavid_v2,
    predict2_video2world_training_2b_droid_success,
    predict2_video2world_training_2b_droid_success_phase2,
    predict2_video2world_training_2b_droid_success_failure,
    predict2_video2world_training_2b_droid_success_failure_tavid_mask,
    predict2_video2world_training_2b_droid_success_v21_tavid_mask,
    predict2_video2world_training_2b_droid_success_v21_instructsam_implicit_mask,
    predict2_video2world_training_2b_droid_success_v21_baseline_nomask_noloss,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context",
    node=predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_instructsam_feature_target_branch",
    node=predict2_video2world_training_2b_droid_success_v21_instructsam_feature_target_branch,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_dense_spatial_target",
    node=predict2_video2world_training_2b_droid_success_v21_dense_spatial_target,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_instructsam_textfree_multisource",
    node=predict2_video2world_training_2b_droid_success_v21_instructsam_textfree_multisource,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_tavid_mask_context",
    node=predict2_video2world_training_2b_droid_success_v21_tavid_mask_context,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_dense_spatial_rawseg",
    node=predict2_video2world_training_2b_droid_success_v21_dense_spatial_rawseg,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_match_ground",
    node=predict2_video2world_training_2b_droid_success_v21_match_ground,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_match_ground_v2",
    node=predict2_video2world_training_2b_droid_success_v21_match_ground_v2,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_match_ground_v3",
    node=predict2_video2world_training_2b_droid_success_v21_match_ground_v3,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_dense_feature_map_target",
    node=predict2_video2world_training_2b_droid_success_v21_dense_feature_map_target,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_maskfree_decoder_dense_target",
    node=predict2_video2world_training_2b_droid_success_v21_maskfree_decoder_dense_target,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_feature_input_channel_target",
    node=predict2_video2world_training_2b_droid_success_v21_feature_input_channel_target,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_feature_control_map_channel_target",
    node=predict2_video2world_training_2b_droid_success_v21_feature_control_map_channel_target,
)

cs.store(
    group="experiment",
    package="_global_",
    name="predict2_video2world_training_2b_droid_success_v21_latent_grounding_decoder_dense_target",
    node=predict2_video2world_training_2b_droid_success_v21_latent_grounding_decoder_dense_target,
)
