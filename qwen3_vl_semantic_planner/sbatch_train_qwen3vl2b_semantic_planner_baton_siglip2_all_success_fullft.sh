#!/usr/bin/env bash
# Baton Stage-1 full fine-tune on the full DROID success strict-holdout training split.

#SBATCH --job-name=q3vl2b-baton-all
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-baton-all-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-baton-all-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT" || exit 2
mkdir -p logs

export DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3}
export PLAN_LABEL_DIR=${PLAN_LABEL_DIR:-$DATASET_ROOT/siglip2_semantic_plan_k6_g9_full}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/user/jhe724/workspace/VLM4WAM/outputs/qwen3vl_semantic_planner/qwen3vl2b_baton_siglip2_all_success_strict_holdout_v3_fullft_8gpu_b2_acc8_gbs128_15000step}
export MAX_STEPS=${MAX_STEPS:-15000}
export BATCH_SIZE=${BATCH_SIZE:-2}
export GRAD_ACCUM=${GRAD_ACCUM:-8}
export LR=${LR:-2e-6}
export HEAD_LR=${HEAD_LR:-1e-4}
export SAVE_STEPS=${SAVE_STEPS:-1000}

exec qwen3_vl_semantic_planner/sbatch_train_qwen3vl2b_semantic_planner_baton_siglip2_fullft.sh
