#!/usr/bin/env bash
# CoVT-style semantic planner: the LM emits a compact continuous-latent bottleneck
# (num_latent_per_keyframe per keyframe) and a decoder reconstructs the dense native
# SigLIP2 grid (k5 x 27x27), mirroring CoVT's DINO branch (1 MHA + 2 FC, feature MSE).
#
# The reconstructed native grid is a drop-in for the oracle plan the current k5/native
# world model consumes -> closes the planner->WM loop without a 3645-token LM sequence
# and without the O(plan_len^2) baton attention.
#
# Requires native k5 labels precomputed on the SAME 10hz dataset the WM trains on.
# Use a SINGLE stride (frame-strides=1) and one window per stem: the planner predicts the
# plan from the first frame alone, so a mixed-stride target would be ambiguous.
#   NUM_KEYFRAMES=5 GRID_SIZE=0 COSMOS_NUM_FRAMES=49 FRAME_STRIDES=1 WINDOWS_PER_STEM=1 \
#   DATASET_ROOT=<10hz> bash scripts/.../sbatch_precompute_siglip2_semantic_plan_labels_window_native_array.sh

#SBATCH --job-name=q3vl2b-covt-native
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-covt-native-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-covt-native-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT" || exit 2
mkdir -p logs

# Match the world model's plan spec exactly: k5, native 27x27 grid, 10hz dataset.
export DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_10hz_320x576}
export PLAN_LABEL_DIR=${PLAN_LABEL_DIR:-$DATASET_ROOT/siglip2_semantic_plan_k5_gnative_cosmos_t49_s1_step24_full}
export OUTPUT_DIR=${OUTPUT_DIR:-/data/user/jhe724/workspace/VLM4WAM/outputs/qwen3vl_semantic_planner/qwen3vl2b_covt_siglip2_native_k5_10hz_8gpu_b2_acc8_gbs128_15000step}
export NUM_KEYFRAMES=${NUM_KEYFRAMES:-5}
export GRID_SIZE=${GRID_SIZE:-27}
export PLAN_HEAD_TYPE=${PLAN_HEAD_TYPE:-covt}
export NUM_LATENT_PER_KEYFRAME=${NUM_LATENT_PER_KEYFRAME:-4}
export SAMPLE_ONE_WINDOW_PER_STEM=${SAMPLE_ONE_WINDOW_PER_STEM:-1}
# Full fine-tune backbone at 1e-5 (2e-6 was too low to adapt), head at 1e-4. Fewer steps
# (8000, ~14 epochs at gbs=128) to limit full-FT forgetting; light weight decay + 400 warmup.
export MAX_STEPS=${MAX_STEPS:-8000}
export BATCH_SIZE=${BATCH_SIZE:-2}
export GRAD_ACCUM=${GRAD_ACCUM:-8}
export LR=${LR:-1e-5}
export HEAD_LR=${HEAD_LR:-1e-4}
export WARMUP_STEPS=${WARMUP_STEPS:-400}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export SAVE_STEPS=${SAVE_STEPS:-1000}

exec qwen3_vl_semantic_planner/sbatch_train_qwen3vl2b_semantic_planner_baton_siglip2_fullft.sh
