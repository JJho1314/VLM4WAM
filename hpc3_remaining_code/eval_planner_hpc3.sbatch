#!/bin/bash
#SBATCH -J eval_plan4b
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o /data/user/jhe724/workspace/VLM4WAM/logs/eval_plan4b_%j.log
set -uo pipefail
R=/data/user/jhe724/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR=$R/.cache/triton_eval TORCHINDUCTOR_CACHE_DIR=$R/.cache/inductor_eval
export LINGBOT_SRC_ROOT=$R/lingbot-vla-v2 UTILS3D_MOGE_PATH=$R/py_deps/utils3d_moge
export PYTHONUNBUFFERED=1
cd $R/VLM4WAM
OUT=$R/VLM4WAM/outputs/eval_lingbot_dino_gbs64
mkdir -p $OUT logs
CKPT=${CKPT:-step_011000}
$HOME/.conda/envs/starVLA/bin/python scripts/qwen3_vl_semantic_planner/evaluate_qwen3vl4b_lingbot_dino_planner.py \
  --checkpoint-dir $R/VLM4WAM/outputs/qwen3vl4b_lingbot_dino_depth_uniform_k5_fullft_gbs64/$CKPT \
  --dataset-root $R/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_10hz_320x576 \
  --dino-teacher-ckpt $R/weights/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth \
  --dino-teacher-config $R/weights/lingbot-vla-v2-6b/dino_video/config.yaml \
  --max-samples 256 --batch-size 2 --num-workers 6 \
  --output-json $OUT/${CKPT}.json
