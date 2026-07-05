#!/usr/bin/env bash
# Qwen3-VL-4B Baton-style Stage-1 training on window-level SigLIP2 semantic plans.
# The dataset groups labels by trajectory stem and samples one random window per
# stem per epoch.

#SBATCH --job-name=q3vl4b-baton-win
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl4b-baton-win-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl4b-baton-win-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT" || exit 2
mkdir -p logs

export MODEL_PATH=${MODEL_PATH:-/data/user/jhe724/workspace/weights/Qwen3-VL-4B-Instruct}
export DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3}
export NUM_KEYFRAMES=${NUM_KEYFRAMES:-16}
export GRID_SIZE=${GRID_SIZE:-9}
export WINDOW_LENGTH=${WINDOW_LENGTH:-93}
export WINDOW_STRIDE=${WINDOW_STRIDE:-24}
export WINDOWS_PER_STEM=${WINDOWS_PER_STEM:-4}
export PLAN_LABEL_DIR=${PLAN_LABEL_DIR:-$DATASET_ROOT/siglip2_semantic_plan_k${NUM_KEYFRAMES}_g${GRID_SIZE}_window_w${WINDOW_LENGTH}_s${WINDOW_STRIDE}_n${WINDOWS_PER_STEM}_full}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/user/jhe724/workspace/VLM4WAM/outputs/qwen3vl_semantic_planner/qwen3vl4b_baton_siglip2_k${NUM_KEYFRAMES}_g${GRID_SIZE}_window_w${WINDOW_LENGTH}_s${WINDOW_STRIDE}_n${WINDOWS_PER_STEM}_sample1traj_fullft_crossattn_8gpu_b1_acc8_gbs64_15000step}

export MAX_STEPS=${MAX_STEPS:-15000}
export BATCH_SIZE=${BATCH_SIZE:-1}
export GRAD_ACCUM=${GRAD_ACCUM:-8}
export NUM_GPUS=${NUM_GPUS:-8}
export SAMPLE_ONE_WINDOW_PER_STEM=${SAMPLE_ONE_WINDOW_PER_STEM:-1}

export LR=${LR:-1e-6}
export HEAD_LR=${HEAD_LR:-1e-4}
export PLAN_HEAD_TYPE=${PLAN_HEAD_TYPE:-baton_crossattn}
export PLAN_HEAD_NUM_HEADS=${PLAN_HEAD_NUM_HEADS:-16}
export PLAN_HEAD_DROPOUT=${PLAN_HEAD_DROPOUT:-0.0}
export SEM_MLP_HIDDEN_SIZE=${SEM_MLP_HIDDEN_SIZE:--1}
export COSINE_LOSS_WEIGHT=${COSINE_LOSS_WEIGHT:-1.0}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export NUM_WORKERS=${NUM_WORKERS:-2}
export FREEZE_VISION=${FREEZE_VISION:-1}
export FREEZE_LM_HEAD=${FREEZE_LM_HEAD:-1}

exec scripts/qwen3_vl_semantic_planner/sbatch_train_qwen3vl2b_semantic_planner_baton_siglip2_fullft.sh
