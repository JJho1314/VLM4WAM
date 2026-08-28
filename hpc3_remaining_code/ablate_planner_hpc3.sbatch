#!/bin/bash
#SBATCH -J ablate_plan4b
#SBATCH -p acd_u
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o /data/user/jhe724/workspace/VLM4WAM/logs/ablate_plan4b_%j.log
set -uo pipefail
R=/data/user/jhe724/workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TRITON_CACHE_DIR=$R/.cache/triton_eval TORCHINDUCTOR_CACHE_DIR=$R/.cache/inductor_eval
export LINGBOT_SRC_ROOT=$R/lingbot-vla-v2 UTILS3D_MOGE_PATH=$R/py_deps/utils3d_moge
export XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
cd $R/VLM4WAM
$HOME/.conda/envs/starVLA/bin/python scripts/qwen3_vl_semantic_planner/ablate_latents_lingbot_dino.py \
  --checkpoint-dir $R/VLM4WAM/outputs/qwen3vl4b_lingbot_dino_depth_uniform_k5_fullft_gbs64/step_011000 \
  --dataset-root $R/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_10hz_320x576 \
  --dino-teacher-ckpt $R/weights/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth \
  --dino-teacher-config $R/weights/lingbot-vla-v2-6b/dino_video/config.yaml \
  --max-samples 64 \
  --output-json $R/VLM4WAM/outputs/eval_lingbot_dino_gbs64/ablate_latents.json
