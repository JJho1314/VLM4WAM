#!/usr/bin/env bash
# VERBATIM RECORD of the outer launcher used for the 2B SigLIP2+DA3 run on pod 30332 (8xH100),
# kept for reproducibility of that experiment's exact hyperparameters. UNLIKE the sibling
# train_dinov3_da3_2b.sh, this is NOT portable: every path below is hardcoded to that pod's
# layout (/data/users/junjie/...). Treat it as a config record, not a script to run elsewhere.
# It sets env vars and delegates to lingbot_dino_4b/train_lingbot_dino_4b.sh, which does the work.
#
# Run wrapper on the pod additionally set: FASTWAM_FRAME_CACHE_DIR=<pod frame cache>,
# NUM_WORKERS=16 (measured-optimal; 24 + deeper prefetch was tried and was 40% SLOWER),
# SAVE_START_STEP=5000 and NCCL_PG_TIMEOUT_SEC=1800 (a transient NCCL stall had killed an
# earlier attempt at step 6414 with no checkpoints saved).
# 2B SigLIP2 + Depth-Anything-3 planner on LIBERO (FastWAM), on Ola H100.
# Sibling of launch_ola_2b_dinov3da3.sh with ONLY the VIDEO teacher swapped DINOv3 -> SigLIP2.
# Everything else mirrors the dinov3 b32a1 baseline, which makes this a clean SigLIP2-vs-DINOv3 A/B:
#   * USE_CURRENT_ALIGNMENT=1 + NUM_KEYFRAMES=1 + KEYFRAME_SCHEME=even_future -> offsets [8]
#     (current-frame alignment + one future keyframe at the horizon end)
#   * INDEPENDENT_MODALITY_TASK_TOKENS=1 -> q64 independent current/future x dino/depth tokens
#   * DA3 stays last_layer (matching the dinov3 last_layer baseline)
# Teacher = siglip2-large-patch16-256 at its NATIVE 256: 256/16 = 16x16 = 256 tokens exactly,
# so there is NO position-embedding interpolation and NO pooling -- the grid already matches the
# DA3 teacher's 16x16. hidden=1024 (24 layers) -> semantic_dim=1024.
# (The so400m-patch14-384 variant is what the WM consumes for semantic_plan; it is label-exact but
# runs 27x27=729 tokens through 27 layers -> 1.97 s/it. This large-256 teacher trades that exact
# WM-space match for far fewer teacher tokens.)
set -euo pipefail
J=/data/users/junjie
ROOT=$J
REPO_ROOT=$J/code/VLM4WAM_k1_zero2_bidir
DATA_ROOT=/data/shared/datasets/libero_fastwam
W2B=$J/vlm4wam_2b/weights
PY=$J/envs/vlm4wam/bin/python
RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=${NUM_GPUS:-8}; BATCH_SIZE=${BATCH_SIZE:-32}; GRAD_ACCUM=${GRAD_ACCUM:-1}
if [[ "$RUN_KIND" == "smoke" ]]; then MAX_STEPS=${MAX_STEPS:-2}; SAVE_STEPS=${SAVE_STEPS:-2}; SAVE_START_STEP=${SAVE_START_STEP:-0}
else MAX_STEPS=${MAX_STEPS:-30000}; SAVE_STEPS=${SAVE_STEPS:-5000}; SAVE_START_STEP=${SAVE_START_STEP:-15000}; fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl2b_siglip2_da3_libero_cur_k1}

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
  FULL_FINETUNE=1 NUM_WORKERS="${NUM_WORKERS:-8}" \
  LR="${LR:-4.24e-5}" HEAD_LR="${HEAD_LR:-4.24e-4}" WARMUP_STEPS="${WARMUP_STEPS:-2500}" \
  PY="$PY" WEIGHTS="$W2B" MODEL_PATH="$MODEL_PATH" \
  HEAD_WARMSTART_CKPT="" SEMANTIC_DIM=0 \
  VIDEO_TARGET_TYPE=siglip2 DEPTH_TARGET_TYPE=da3 \
  SIGLIP2_MODEL_DIR="$SIGLIP2_MODEL_DIR" SIGLIP2_GRID_SIZE="${SIGLIP2_GRID_SIZE:-16}" \
  SIGLIP2_INPUT_SIZE="${SIGLIP2_INPUT_SIZE:-256}" \
  DA3_CKPT_DIR="$DA3_CKPT_DIR" DA3_CODE_ROOT="$DA3_CODE_ROOT" DA3_PROCESS_RES=224 \
  DA3_ALIGN_STRATEGY=last_layer \
  USE_CURRENT_ALIGNMENT=1 INDEPENDENT_MODALITY_TASK_TOKENS=1 \
  NUM_KEYFRAMES="${NUM_KEYFRAMES:-1}" KEYFRAME_SCHEME="${KEYFRAME_SCHEME:-even_future}" \
  LINGBOT_SRC_ROOT="$ROOT/code/lingbot-vla-v2" UTILS3D_MOGE_PATH="$ROOT/py_deps/utils3d_moge" \
  FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  FASTWAM_DATASET_DIRS="$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot" \
  FASTWAM_FRAME_CACHE_DIR="${FASTWAM_FRAME_CACHE_DIR:-$ROOT/data/frame_cache/libero}" \
  FASTWAM_TEXT_EMBEDDING_CACHE_DIR="$ROOT/data/libero_qwen" \
  FASTWAM_PRETRAINED_NORM_STATS="$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
