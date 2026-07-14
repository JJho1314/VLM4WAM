#!/usr/bin/env bash
# CoVT planner with ONLINE cosmos-style episode sampling + LATE keyframes.
#
# Aligns the planner's training distribution with the world model on both axes the
# fixed-label run could not:
#   1. window sampling — a fresh random 49f stride-1 window per stem per epoch from
#      frame_ranges.json (same episode-sampling idea as the WM's online training), so the
#      planner also learns mid-episode starts (needed for closed-loop replanning);
#   2. keyframe scheme — LATE [1,21,32,41,48] to match the recommended late WM
#      (see eval verdict: late fixes end-state placement).
# SigLIP2 targets are encoded on the fly with the offline builder's exact encode path;
# no precomputed label dir is needed.

#SBATCH --job-name=q3vl2b-covt-online-late
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-covt-online-late-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-covt-online-late-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT" || exit 2
mkdir -p logs

export DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_10hz_320x576}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/user/jhe724/workspace/VLM4WAM/outputs/qwen3vl_semantic_planner/qwen3vl2b_covt_online_late_k5_10hz_8gpu_b2_acc8_gbs128_8000step}
export ONLINE_PLAN_LABELS=1
export SIGLIP2_ENCODER_PATH=${SIGLIP2_ENCODER_PATH:-/data/user/jhe724/workspace/weights/siglip2-so400m-patch14-384}
export KEYFRAME_SCHEME=${KEYFRAME_SCHEME:-late}
export KEYFRAME_GAMMA=${KEYFRAME_GAMMA:-0.6}
export SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-49}
export ONLINE_GRID_SIZE=${ONLINE_GRID_SIZE:-0}
export SEMANTIC_DIM=${SEMANTIC_DIM:-1152}
export NUM_KEYFRAMES=${NUM_KEYFRAMES:-5}
export GRID_SIZE=${GRID_SIZE:-27}
export PLAN_HEAD_TYPE=${PLAN_HEAD_TYPE:-covt}
export NUM_LATENT_PER_KEYFRAME=${NUM_LATENT_PER_KEYFRAME:-4}
export MAX_STEPS=${MAX_STEPS:-8000}
export BATCH_SIZE=${BATCH_SIZE:-2}
export GRAD_ACCUM=${GRAD_ACCUM:-8}
export LR=${LR:-1e-5}
export HEAD_LR=${HEAD_LR:-1e-4}
export WARMUP_STEPS=${WARMUP_STEPS:-400}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export SAVE_STEPS=${SAVE_STEPS:-1000}
# online mode loads 6 video frames per sample instead of 1 -> more IO workers
export NUM_WORKERS=${NUM_WORKERS:-6}

exec qwen3_vl_semantic_planner/sbatch_train_qwen3vl2b_semantic_planner_baton_siglip2_fullft.sh
