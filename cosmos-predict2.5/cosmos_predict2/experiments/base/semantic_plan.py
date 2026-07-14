# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Baton-style semantic-plan post-training configs for Cosmos Video2World."""

import os

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import (
    VideoDataset,
    get_generic_dataloader,
    get_sampler,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]

DATASET_ROOT = os.environ.get(
    "DATASET_ROOT",
    "/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3",
)
CHECKPOINT_LOAD_PATH = os.environ.get(
    "CHECKPOINT_LOAD_PATH",
    "/data/user/jhe724/workspace/weights/Cosmos-Predict2.5-2B/base/post-trained/"
    "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
)

NUM_FRAMES = int(os.environ.get("COSMOS_NUM_FRAMES", "93"))
# Wan2.1 VAE: 4x temporal compression with the first frame kept, so 93f -> 24, 49f -> 13.
STATE_T = int(os.environ.get("COSMOS_STATE_T", str((NUM_FRAMES - 1) // 4 + 1)))
VIDEO_HEIGHT = int(os.environ.get("VIDEO_HEIGHT", "320"))
VIDEO_WIDTH = int(os.environ.get("VIDEO_WIDTH", "576"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
MAX_ITER = int(os.environ.get("MAX_ITER", "3000"))
GRAD_ACCUM_ITER = int(os.environ.get("GRAD_ACCUM_ITER", "8"))
SAVE_ITER = int(os.environ.get("SAVE_ITER", "500"))
TRAIN_SAMPLE_EVERY = int(os.environ.get("TRAIN_SAMPLE_EVERY", "0"))
LOGGING_ITER = int(os.environ.get("LOGGING_ITER", "20"))

SEMANTIC_PLAN_DIM = int(os.environ.get("SEMANTIC_PLAN_DIM", "1152"))
SEMANTIC_PLAN_NUM_KEYFRAMES = int(os.environ.get("SEMANTIC_PLAN_NUM_KEYFRAMES", "6"))
SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES = int(
    os.environ.get("SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES", "16")
)
SEMANTIC_PLAN_SPATIAL_GRID = int(os.environ.get("SEMANTIC_PLAN_SPATIAL_GRID", "9"))
SEMANTIC_PLAN_MAX_TOKENS = int(
    os.environ.get(
        "SEMANTIC_PLAN_MAX_TOKENS",
        str(SEMANTIC_PLAN_NUM_KEYFRAMES * SEMANTIC_PLAN_SPATIAL_GRID * SEMANTIC_PLAN_SPATIAL_GRID),
    )
)
_SEMANTIC_PLAN_SOURCE_TOKENS = (
    SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES * SEMANTIC_PLAN_SPATIAL_GRID * SEMANTIC_PLAN_SPATIAL_GRID
    if SEMANTIC_PLAN_SPATIAL_GRID > 0
    else 0
)
SEMANTIC_PLAN_DATASET_MAX_TOKENS = int(
    os.environ.get("SEMANTIC_PLAN_DATASET_MAX_TOKENS", str(_SEMANTIC_PLAN_SOURCE_TOKENS))
)
SEMANTIC_PLAN_FRAME_STRIDES = os.environ.get("SEMANTIC_PLAN_FRAME_STRIDES", "1,2,3")
SEMANTIC_PLAN_WINDOW_STRIDE = int(os.environ.get("SEMANTIC_PLAN_WINDOW_STRIDE", "24"))
SEMANTIC_PLAN_GRID_TAG = "gnative" if SEMANTIC_PLAN_SPATIAL_GRID <= 0 else f"g{SEMANTIC_PLAN_SPATIAL_GRID}"
SEMANTIC_PLAN_STRIDE_TAG = SEMANTIC_PLAN_FRAME_STRIDES.replace(",", "")
SEMANTIC_PLAN_DIR = os.environ.get(
    "SEMANTIC_PLAN_DIR",
    os.path.join(
        DATASET_ROOT,
        (
            f"siglip2_semantic_plan_k{SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES}_"
            f"{SEMANTIC_PLAN_GRID_TAG}_cosmos_t{NUM_FRAMES}_s{SEMANTIC_PLAN_STRIDE_TAG}_"
            f"step{SEMANTIC_PLAN_WINDOW_STRIDE}_full"
        ),
    ),
)
SEMANTIC_PLAN_MANIFEST = os.environ.get("SEMANTIC_PLAN_MANIFEST", "manifest*.jsonl")
VAE_LATENT_DIR = os.environ.get("VAE_LATENT_DIR", "")
VAE_LATENT_MANIFEST = os.environ.get("VAE_LATENT_MANIFEST", "window_manifest*.jsonl")
SEMANTIC_PLAN_HIDDEN_DIM = int(os.environ.get("SEMANTIC_PLAN_HIDDEN_DIM", "2048"))
SEMANTIC_PLAN_COORD_HIDDEN_DIM = int(os.environ.get("SEMANTIC_PLAN_COORD_HIDDEN_DIM", "256"))
SEMANTIC_PLAN_DROPOUT_PROB = float(os.environ.get("SEMANTIC_PLAN_DROPOUT_PROB", "0.15"))
# Online encoding: skip precomputed .pt features and encode SigLIP2 plans on the fly from
# the training video window (manifest still defines windows + VAE latents). Enables native
# grids (SEMANTIC_PLAN_SPATIAL_GRID=0 -> 27x27) without terabytes of label storage.
SEMANTIC_PLAN_ONLINE = os.environ.get("SEMANTIC_PLAN_ONLINE", "0") == "1"
SEMANTIC_PLAN_ONLINE_ENCODER_PATH = os.environ.get(
    "SEMANTIC_PLAN_ONLINE_ENCODER_PATH",
    "/data/user/jhe724/workspace/VLM4WAM/third_party/siglip2-so400m-patch14-384",
)
SEMANTIC_PLAN_ONLINE_NUM_KEYFRAMES = int(
    os.environ.get("SEMANTIC_PLAN_ONLINE_NUM_KEYFRAMES", str(SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES))
)
# Ctrl-World-style episode sampling: windows drawn fully at random inside __getitem__
# (range picked proportional to length, random stride, random start) from the motion ranges
# in frame_ranges.json. No window manifest / .pt features / VAE latent cache are used;
# combine with SEMANTIC_PLAN_ONLINE=1.
SEMANTIC_PLAN_EPISODE_SAMPLING = os.environ.get("SEMANTIC_PLAN_EPISODE_SAMPLING", "0") == "1"
EPISODE_SAMPLES_PER_EPOCH = int(os.environ.get("EPISODE_SAMPLES_PER_EPOCH", "0"))
EPISODE_FRAME_STRIDES = [
    int(item) for item in SEMANTIC_PLAN_FRAME_STRIDES.split(",") if item.strip()
]
SEMANTIC_PLAN_BLOCKS = [
    int(item)
    for item in os.environ.get("SEMANTIC_PLAN_BLOCKS", ",".join(str(i) for i in range(28))).split(",")
    if item.strip()
]

semantic_plan_droid_dataset = L(VideoDataset)(
    dataset_dir=DATASET_ROOT,
    num_frames=NUM_FRAMES,
    video_size=(VIDEO_HEIGHT, VIDEO_WIDTH),
    caption_format=os.environ.get("CAPTION_FORMAT", "auto"),
    prompt_type=os.environ.get("PROMPT_TYPE", None),
    semantic_plan_dir=None if SEMANTIC_PLAN_EPISODE_SAMPLING else SEMANTIC_PLAN_DIR,
    semantic_plan_skip_features=SEMANTIC_PLAN_ONLINE,
    semantic_plan_default_to_zero=os.environ.get("SEMANTIC_PLAN_DEFAULT_TO_ZERO", "0") == "1",
    semantic_plan_dim=SEMANTIC_PLAN_DIM,
    semantic_plan_max_tokens=SEMANTIC_PLAN_DATASET_MAX_TOKENS,
    semantic_plan_manifest=(
        None
        if SEMANTIC_PLAN_EPISODE_SAMPLING or SEMANTIC_PLAN_MANIFEST.lower() == "none"
        else SEMANTIC_PLAN_MANIFEST
    ),
    vae_latent_dir=None if not VAE_LATENT_DIR or VAE_LATENT_DIR.lower() == "none" else VAE_LATENT_DIR,
    vae_latent_manifest=None if VAE_LATENT_MANIFEST.lower() == "none" else VAE_LATENT_MANIFEST,
    episode_sampling=SEMANTIC_PLAN_EPISODE_SAMPLING,
    frame_ranges_path=os.environ.get("FRAME_RANGES_JSON", os.path.join(DATASET_ROOT, "frame_ranges.json")),
    episode_frame_strides=EPISODE_FRAME_STRIDES,
    episode_samples_per_epoch=EPISODE_SAMPLES_PER_EPOCH,
)

semantic_plan_droid_dataloader = L(get_generic_dataloader)(
    dataset=semantic_plan_droid_dataset,
    sampler=L(get_sampler)(dataset=semantic_plan_droid_dataset),
    batch_size=BATCH_SIZE,
    drop_last=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=int(os.environ.get("PREFETCH_FACTOR", "2")),
    persistent_workers=os.environ.get("PERSISTENT_WORKERS", "1") == "1",
)

predict2_video2world_training_2b_droid_semantic_plan_320x576_93f = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "mock"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project=os.environ.get("WANDB_PROJECT", "cosmos_predict_v2p5"),
        group=os.environ.get("WANDB_GROUP", "semantic_plan_video2world"),
        name=os.environ.get("JOB_NAME", "2b_droid_semantic_plan_320x576_93f"),
        wandb_mode=os.environ.get("WANDB_MODE", "online"),
    ),
    dataloader_train=semantic_plan_droid_dataloader,
    checkpoint=dict(
        save_iter=SAVE_ITER,
        load_path=CHECKPOINT_LOAD_PATH,
        load_training_state=False,
        strict_resume=False,
        load_ema_to_reg=True,
        load_from_object_store=dict(
            enabled=False,
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    optimizer=dict(
        lr=float(os.environ.get("LR", str(2 ** (-14.5)))),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.001")),
    ),
    scheduler=dict(
        f_max=[float(os.environ.get("SCHED_F_MAX", "0.5"))],
        f_min=[float(os.environ.get("SCHED_F_MIN", "0.2"))],
        warm_up_steps=[int(os.environ.get("WARMUP_STEPS", "500"))],
        cycle_lengths=[int(os.environ.get("SCHED_CYCLE_LENGTH", "100000"))],
    ),
    trainer=dict(
        straggler_detection=dict(enabled=False),
        max_iter=MAX_ITER,
        grad_accum_iter=GRAD_ACCUM_ITER,
        logging_iter=LOGGING_ITER,
        validation_iter=int(os.environ.get("VALIDATION_ITER", "1000000000")),
        run_validation=False,
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(hit_thres=200, save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(every_n=TRAIN_SAMPLE_EVERY, save_s3=False),
            every_n_sample_ema=dict(every_n=TRAIN_SAMPLE_EVERY, save_s3=False),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("CONTEXT_PARALLEL_SIZE", "1")),
    ),
    model=dict(
        config=dict(
            state_t=STATE_T,
            min_num_conditional_frames=1,
            max_num_conditional_frames=1,
            conditional_frames_probs=None,
            semantic_plan_dropout_prob=SEMANTIC_PLAN_DROPOUT_PROB,
            semantic_plan_online_encoder_path=(
                SEMANTIC_PLAN_ONLINE_ENCODER_PATH if SEMANTIC_PLAN_ONLINE else ""
            ),
            semantic_plan_online_num_keyframes=SEMANTIC_PLAN_ONLINE_NUM_KEYFRAMES,
            semantic_plan_online_grid_size=SEMANTIC_PLAN_SPATIAL_GRID,
            net=dict(
                semantic_plan_context=True,
                semantic_plan_in_dim=SEMANTIC_PLAN_DIM,
                semantic_plan_hidden_dim=SEMANTIC_PLAN_HIDDEN_DIM,
                semantic_plan_max_tokens=SEMANTIC_PLAN_MAX_TOKENS,
                semantic_plan_num_keyframes=SEMANTIC_PLAN_NUM_KEYFRAMES,
                semantic_plan_source_num_keyframes=SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES,
                semantic_plan_spatial_grid=SEMANTIC_PLAN_SPATIAL_GRID,
                semantic_plan_coord_hidden_dim=SEMANTIC_PLAN_COORD_HIDDEN_DIM,
                semantic_plan_use_rope=os.environ.get("SEMANTIC_PLAN_USE_ROPE", "1") == "1",
                semantic_plan_cross_attention_blocks=SEMANTIC_PLAN_BLOCKS,
            ),
        ),
    ),
)

cs = ConfigStore.instance()

for _item in [
    predict2_video2world_training_2b_droid_semantic_plan_320x576_93f,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
