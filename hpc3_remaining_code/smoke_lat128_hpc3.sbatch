#!/bin/bash
#SBATCH -J smoke_lat128
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o /data/user/jhe724/workspace/VLM4WAM/logs/smoke_lat128_%j.log
set -uo pipefail
R=/data/user/jhe724/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR=$R/.cache/triton TORCHINDUCTOR_CACHE_DIR=$R/.cache/inductor
cd $R/VLM4WAM
NUM_GPUS=1 BATCH_SIZE=2 GRAD_ACCUM=1 MAX_STEPS=2 SAVE_STEPS=1000 LOG_STEPS=1 \
  FULL_FINETUNE=1 USE_DEPTH=1 NUM_WORKERS=2 \
  NUM_LATENT_PER_KEYFRAME=128 NUM_HEAD_LATENT_PER_KEYFRAME=128 \
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
  OUTPUT_DIR=$R/VLM4WAM/outputs/smoke_lat128 \
  bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
