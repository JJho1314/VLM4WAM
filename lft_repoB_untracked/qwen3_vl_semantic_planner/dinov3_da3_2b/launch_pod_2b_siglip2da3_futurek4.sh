#!/usr/bin/env bash
# 2B SigLIP2 + Depth-Anything-3 planner on LIBERO (FastWAM), pod 30332 (8xH100).
# FUTURE-ONLY, K=4 variant of the current+K=1 baseline. Predicts 4 future keyframes of both
# the SigLIP2 (video) and DA3 (depth) features, no current-frame alignment.
#   * USE_CURRENT_ALIGNMENT=0  -> future-only; the current/future 4-branch task-token path is off,
#     so plan latents come from the per-keyframe shared+private latents (32+32) instead. This also
#     means INDEPENDENT_MODALITY_TASK_TOKENS and DA3 wsa_multilayer are NOT usable (both require
#     current alignment) -> DA3 stays last_layer.
#   * NUM_KEYFRAMES=4 + KEYFRAME_SCHEME=even_future -> offsets [2,4,6,8] (uniform over the 8 future
#     frames of the length-9 window). target tokens = 4 * 16^2 = 1024.
# Teacher = siglip2-large-patch16-256 at native 256 (16x16=256 tok, no interp/pool), semantic_dim=1024.
# Everything else mirrors the k1 baseline (global batch 256, LR 4.24e-5, 30000 steps). Runs in the
# clean VLM4WAM_git checkout (new top-level layout, matches the workspace source-of-truth).
set -euo pipefail
J=/data/users/junjie
ROOT=$J
REPO_ROOT=$J/code/VLM4WAM_git
DATA_ROOT=/data/shared/datasets/libero_fastwam
W2B=$J/vlm4wam_2b/weights
PY=$J/envs/vlm4wam/bin/python
RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=${NUM_GPUS:-8}; BATCH_SIZE=${BATCH_SIZE:-32}; GRAD_ACCUM=${GRAD_ACCUM:-1}
if [[ "$RUN_KIND" == "smoke" ]]; then MAX_STEPS=${MAX_STEPS:-2}; SAVE_STEPS=${SAVE_STEPS:-2}; SAVE_START_STEP=${SAVE_START_STEP:-0}
else MAX_STEPS=${MAX_STEPS:-30000}; SAVE_STEPS=${SAVE_STEPS:-5000}; SAVE_START_STEP=${SAVE_START_STEP:-5000}; fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl2b_siglip2_da3_libero_future_k4}

MODEL_PATH=$W2B/Qwen3-VL-2B-Instruct
SIGLIP2_MODEL_DIR=${SIGLIP2_MODEL_DIR:-$W2B/siglip2-large-patch16-256}
DA3_CKPT_DIR=$W2B/DA3-LARGE-1.1
DA3_CODE_ROOT=$J/vlm4wam_2b/code/Depth-Anything-3

for p in "$PY" "$MODEL_PATH" "$SIGLIP2_MODEL_DIR/config.json" "$DA3_CKPT_DIR/config.json" \
  "$DA3_CODE_ROOT/src/depth_anything_3/api.py" \
  "$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" "$ROOT/data/libero_qwen" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" "$DATA_ROOT/libero_10_no_noops_lerobot"; do
  [[ -e "$p" ]] || { echo "ERROR missing: $p" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs" "$ROOT/cache/triton" "$ROOT/cache/inductor"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PLANNER_WANDB=${PLANNER_WANDB:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$ROOT/cache/triton TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "$REPO_ROOT"
exec env \
  NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
  EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-256}" MAX_STEPS="$MAX_STEPS" \
  SAVE_STEPS="$SAVE_STEPS" SAVE_START_STEP="$SAVE_START_STEP" \
  FULL_FINETUNE=1 NUM_WORKERS="${NUM_WORKERS:-16}" \
  LR="${LR:-4.24e-5}" HEAD_LR="${HEAD_LR:-4.24e-4}" WARMUP_STEPS="${WARMUP_STEPS:-2500}" \
  PY="$PY" WEIGHTS="$W2B" MODEL_PATH="$MODEL_PATH" \
  HEAD_WARMSTART_CKPT="" SEMANTIC_DIM=0 \
  VIDEO_TARGET_TYPE=siglip2 DEPTH_TARGET_TYPE=da3 \
  SIGLIP2_MODEL_DIR="$SIGLIP2_MODEL_DIR" SIGLIP2_GRID_SIZE="${SIGLIP2_GRID_SIZE:-16}" \
  SIGLIP2_INPUT_SIZE="${SIGLIP2_INPUT_SIZE:-256}" \
  DA3_CKPT_DIR="$DA3_CKPT_DIR" DA3_CODE_ROOT="$DA3_CODE_ROOT" DA3_PROCESS_RES=224 \
  DA3_ALIGN_STRATEGY=last_layer \
  USE_CURRENT_ALIGNMENT=0 INDEPENDENT_MODALITY_TASK_TOKENS=0 \
  NUM_KEYFRAMES="${NUM_KEYFRAMES:-4}" KEYFRAME_SCHEME="${KEYFRAME_SCHEME:-even_future}" \
  LINGBOT_SRC_ROOT="$ROOT/code/lingbot-vla-v2" UTILS3D_MOGE_PATH="$ROOT/py_deps/utils3d_moge" \
  FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  FASTWAM_DATASET_DIRS="$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot" \
  FASTWAM_FRAME_CACHE_DIR="${FASTWAM_FRAME_CACHE_DIR:-$ROOT/data/frame_cache/libero}" \
  FASTWAM_TEXT_EMBEDDING_CACHE_DIR="$ROOT/data/libero_qwen" \
  FASTWAM_PRETRAINED_NORM_STATS="$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
