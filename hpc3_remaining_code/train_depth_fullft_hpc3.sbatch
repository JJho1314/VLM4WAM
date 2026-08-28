#!/bin/bash
# Real 4B lingbot-DINO + future-DEPTH training on HPC3 (8xH100, full-FT).
#   video MSE (DINO-video teacher) + depth smooth_L1 (MoGe-2 -> MoRGBD), both future-only.
#   warm-start: video head + depth head from the extracted align-head package.
#SBATCH -J plan4b_dino_depth
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 2-00:00:00
#SBATCH -o /data/user/jhe724/workspace/VLM4WAM/logs/plan4b_dino_depth_%j.log
set -uo pipefail
R=/data/user/jhe724/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# HPC3: /data/user/jhe724/bin/gcc is broken (cc1 missing) -> force the working system gcc so the
# DINO teachers flex_attention (torch.compile->inductor->triton) can build its cuda_utils helper.
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR=$R/.cache/triton TORCHINDUCTOR_CACHE_DIR=$R/.cache/inductor
mkdir -p $R/VLM4WAM/logs $R/.cache/triton $R/.cache/inductor
cd $R/VLM4WAM
NUM_GPUS=8 BATCH_SIZE=2 GRAD_ACCUM=4 MAX_STEPS=11000 SAVE_STEPS=1000 LOG_STEPS=10 \
  FULL_FINETUNE=1 USE_DEPTH=1 NUM_WORKERS=4 \
  LR=3e-5 HEAD_LR=3e-4 WARMUP_STEPS=1000 \
  PY=$HOME/.conda/envs/starVLA/bin/python \
  WEIGHTS=$R/weights \
  MODEL_PATH=$R/weights/Qwen3-VL-4B-lingbot-vlm \
  LINGBOT_6B=$R/weights/lingbot-vla-v2-6b \
  HEAD_WARMSTART_CKPT=$R/weights/lingbot_align_heads_warmstart \
  DEPTH_MOGE_PATH=$R/weights/moge-2-vitb-normal/model.pt \
  DEPTH_MORGBD_PATH=$R/weights/lingbot-vla-v2-6b/depth/model.pt \
  DATASET_ROOT=$R/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_10hz_320x576 \
  LINGBOT_SRC_ROOT=$R/lingbot-vla-v2 \
  UTILS3D_MOGE_PATH=$R/py_deps/utils3d_moge \
  OUTPUT_DIR=$R/VLM4WAM/outputs/qwen3vl4b_lingbot_dino_depth_uniform_k5_fullft_gbs64 \
  bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
