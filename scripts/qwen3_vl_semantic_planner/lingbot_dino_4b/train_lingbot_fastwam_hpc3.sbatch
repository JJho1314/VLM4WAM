#!/usr/bin/env bash
#SBATCH -J vlmp_zero2_q64
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

WORKSPACE=${WORKSPACE:-/data/user/jhe724/workspace}
REPO_ROOT=${REPO_ROOT:-$WORKSPACE/VLM4WAM_lingbot_zero2_20260713}
DATA_ROOT=${DATA_ROOT:-$WORKSPACE/datasets/LIBERO-fastwam}
TEXT_CACHE=${TEXT_CACHE:-$WORKSPACE/datasets/libero_qwen}
NORM_STATS=${NORM_STATS:-$WORKSPACE/datasets/LIBERO-fastwam_meta/dataset_stats.json}
WEIGHTS=${WEIGHTS:-$WORKSPACE/weights}
PY=${PY:-$HOME/.conda/envs/starVLA/bin/python}
RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=8
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
if [[ "$RUN_KIND" == "smoke" ]]; then
  MAX_STEPS=${MAX_STEPS:-2}
  SAVE_STEPS=${SAVE_STEPS:-2}
else
  MAX_STEPS=${MAX_STEPS:-12000}
  SAVE_STEPS=${SAVE_STEPS:-1000}
fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl4b_lingbot_independent_q64_zero2_k1_b${BATCH_SIZE}a${GRAD_ACCUM}_${SLURM_JOB_ID}}

for path in \
  "$PY" \
  "$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  "$WEIGHTS/lingbot_align_heads_warmstart/model.safetensors.index.json" \
  "$WEIGHTS/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
  "$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  "$WEIGHTS/moge-2-vitb-normal/model.pt" \
  "$NORM_STATS" \
  "$TEXT_CACHE" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" \
  "$DATA_ROOT/libero_object_no_noops_lerobot" \
  "$DATA_ROOT/libero_goal_no_noops_lerobot" \
  "$DATA_ROOT/libero_10_no_noops_lerobot"; do
  [[ -e "$path" ]] || { echo "ERROR: missing required path: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs" "$WORKSPACE/.cache/triton" "$WORKSPACE/.cache/inductor"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PLANNER_WANDB=${PLANNER_WANDB:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$WORKSPACE/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$WORKSPACE/.cache/inductor
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "$REPO_ROOT"
exec env \
  NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
  EXPECTED_GLOBAL_BATCH=128 MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  FULL_FINETUNE=1 NUM_WORKERS=4 LR=3e-5 HEAD_LR=3e-4 WARMUP_STEPS=1000 \
  PY="$PY" WEIGHTS="$WEIGHTS" \
  MODEL_PATH="$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  LINGBOT_6B="$WEIGHTS/lingbot-vla-v2-6b" \
  HEAD_WARMSTART_CKPT="$WEIGHTS/lingbot_align_heads_warmstart" \
  DEPTH_MOGE_PATH="$WEIGHTS/moge-2-vitb-normal/model.pt" \
  DEPTH_MORGBD_PATH="$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  LINGBOT_SRC_ROOT="$WORKSPACE/lingbot-vla-v2" \
  UTILS3D_MOGE_PATH="$WORKSPACE/py_deps/utils3d_moge" \
  FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  FASTWAM_DATASET_DIRS="$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot" \
  FASTWAM_TEXT_EMBEDDING_CACHE_DIR="$TEXT_CACHE" \
  FASTWAM_PRETRAINED_NORM_STATS="$NORM_STATS" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
