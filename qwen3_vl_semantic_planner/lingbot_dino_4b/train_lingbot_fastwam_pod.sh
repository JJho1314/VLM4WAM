#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/nas/junjie}
REPO_ROOT=${REPO_ROOT:-$ROOT/code/VLM4WAM_k1_zero2_20260713}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/LIBERO-fastwam}
WEIGHTS=${WEIGHTS:-$ROOT/weights}
PY=${PY:-/opt/conda/envs/vlm4wam/bin/python}
RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
SAVE_START_STEP=${SAVE_START_STEP:-0}
if [[ "$RUN_KIND" == "smoke" ]]; then
  MAX_STEPS=${MAX_STEPS:-2}
  SAVE_STEPS=${SAVE_STEPS:-2}
else
  MAX_STEPS=${MAX_STEPS:-12000}
  SAVE_STEPS=${SAVE_STEPS:-1000}
fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl4b_lingbot_independent_q64_zero2_k1_b${BATCH_SIZE}a${GRAD_ACCUM}}

for path in \
  "$PY" \
  "$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  "$WEIGHTS/lingbot_align_heads_warmstart/model.safetensors.index.json" \
  "$WEIGHTS/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
  "$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  "$WEIGHTS/moge-2-vitb-normal/model.pt" \
  "$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" \
  "$ROOT/data/libero_qwen" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" \
  "$DATA_ROOT/libero_object_no_noops_lerobot" \
  "$DATA_ROOT/libero_goal_no_noops_lerobot" \
  "$DATA_ROOT/libero_10_no_noops_lerobot"; do
  [[ -e "$path" ]] || { echo "ERROR: missing required path: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs" "$ROOT/cache/triton" "$ROOT/cache/inductor"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PLANNER_WANDB=${PLANNER_WANDB:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "$REPO_ROOT"
exec env \
  NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
  EXPECTED_GLOBAL_BATCH=128 MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  SAVE_START_STEP="$SAVE_START_STEP" \
  FULL_FINETUNE=1 NUM_WORKERS=4 LR=3e-5 HEAD_LR=3e-4 WARMUP_STEPS=1000 \
  BIDIRECTIONAL_PLAN_ATTN="${BIDIRECTIONAL_PLAN_ATTN:-0}" \
  PY="$PY" WEIGHTS="$WEIGHTS" \
  MODEL_PATH="$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  LINGBOT_6B="$WEIGHTS/lingbot-vla-v2-6b" \
  HEAD_WARMSTART_CKPT="$WEIGHTS/lingbot_align_heads_warmstart" \
  DEPTH_MOGE_PATH="$WEIGHTS/moge-2-vitb-normal/model.pt" \
  DEPTH_MORGBD_PATH="$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  LINGBOT_SRC_ROOT="$ROOT/code/lingbot-vla-v2" \
  UTILS3D_MOGE_PATH="$ROOT/py_deps/utils3d_moge" \
  FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  FASTWAM_DATASET_DIRS="$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot" \
  FASTWAM_TEXT_EMBEDDING_CACHE_DIR="$ROOT/data/libero_qwen" \
  FASTWAM_PRETRAINED_NORM_STATS="$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
