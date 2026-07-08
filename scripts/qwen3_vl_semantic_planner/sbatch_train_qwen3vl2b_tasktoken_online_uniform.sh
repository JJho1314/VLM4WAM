#!/bin/bash
#SBATCH --job-name=q3vl2b-tasktoken-online-uniform
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-tasktoken-online-uniform-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-tasktoken-online-uniform-%j.err

# Independent launcher for the lingbot-vla-v2-style TASKTOKEN planner variant.
# Points TRAIN_SCRIPT at the standalone train_qwen3vl_tasktoken_planner.py — it does NOT touch the
# SigLIP CoVT production script/launcher. Same online-uniform-k5 recipe as the covt run so the two
# are directly comparable (same target = SigLIP, so the current WM still consumes the plan).
set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT" || exit 2
mkdir -p logs

# --- the one line that makes this independent: run the tasktoken script, not the CoVT one ---
export TRAIN_SCRIPT=${TRAIN_SCRIPT:-scripts/qwen3_vl_semantic_planner/train_qwen3vl_tasktoken_planner.py}

export DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_10hz_320x576}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/user/jhe724/workspace/VLM4WAM/outputs/qwen3vl_semantic_planner/qwen3vl2b_tasktoken_online_uniform_k5_10hz_8gpu_b2_acc8_gbs128_16000step}
# base VLM: keep 2B for an apples-to-apples A/B vs covt; point at Qwen3-VL-4B to test the 4B idea.
export MODEL_PATH=${MODEL_PATH:-/data/user/jhe724/workspace/weights/Qwen3-VL-2B-Instruct}

export ONLINE_PLAN_LABELS=1
export SIGLIP2_ENCODER_PATH=${SIGLIP2_ENCODER_PATH:-/data/user/jhe724/workspace/weights/siglip2-so400m-patch14-384}
export KEYFRAME_SCHEME=${KEYFRAME_SCHEME:-uniform}
export KEYFRAME_GAMMA=${KEYFRAME_GAMMA:-0.6}
export SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-49}
export ONLINE_GRID_SIZE=${ONLINE_GRID_SIZE:-0}
export SEMANTIC_DIM=${SEMANTIC_DIM:-1152}
export NUM_KEYFRAMES=${NUM_KEYFRAMES:-5}
export GRID_SIZE=${GRID_SIZE:-27}
export PLAN_HEAD_TYPE=${PLAN_HEAD_TYPE:-tasktoken}
export NUM_LATENT_PER_KEYFRAME=${NUM_LATENT_PER_KEYFRAME:-4}
export MAX_STEPS=${MAX_STEPS:-16000}
export BATCH_SIZE=${BATCH_SIZE:-2}
export GRAD_ACCUM=${GRAD_ACCUM:-8}
export LR=${LR:-1e-5}
export HEAD_LR=${HEAD_LR:-1e-4}
export WARMUP_STEPS=${WARMUP_STEPS:-400}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export NUM_WORKERS=${NUM_WORKERS:-6}

exec scripts/qwen3_vl_semantic_planner/sbatch_train_qwen3vl2b_semantic_planner_baton_siglip2_fullft.sh
