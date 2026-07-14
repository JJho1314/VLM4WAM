#!/bin/bash
# Launcher for the 2B DINOv3 + Depth-Anything-3 planner line.
# Same query/head as the 4B lingbot-DINO line (LingbotDinoPlanHead / TaskTokenResampler), but:
#   base VLM  = Qwen3-VL-2B-Instruct (stock; sem_plan tokens added by the trainer)
#   video tgt = Meta DINOv3 (ViT-H+/16, 256 tok/kf, dim 1280)   [--video-target-type dinov3]
#   depth tgt = Depth-Anything-3 encoder (ViT-L, 256 tok/kf, dim 2048) [--depth-target-type da3]
# No lingbot head warm-start (different teacher space -> heads trained from scratch).
#
# Required: DATASET_ROOT (semantic-plan dataset with keyframe images + frame_ranges.json)
# Smoke:    NUM_GPUS=1 BATCH_SIZE=1 MAX_STEPS=2 FULL_FINETUNE=0 DATASET_ROOT=<...> bash train_dinov3_da3_2b.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANNER_DIR="$(dirname "$HERE")"                       # qwen3_vl_semantic_planner
REPO_ROOT="$(cd "$PLANNER_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 2

PY=${PY:-/data/LFT-W02_data/.conda/envs/starVLA/bin/python}
NUM_GPUS=${NUM_GPUS:-2}

WEIGHTS=${WEIGHTS:-/data/LFT-W02_data/junjie/weights}
MODEL_PATH=${MODEL_PATH:-$WEIGHTS/Qwen3-VL-2B-Instruct}          # stock 2B base

# --- teachers (frozen alignment targets) ---
DINOV3_MODEL_DIR=${DINOV3_MODEL_DIR:-/data/LFT-W02_data/junjie/VLA_WM/LAST-ViT/weights/dinov3_vith16plus/facebook/dinov3-vith16plus-pretrain-lvd1689m}
DA3_CKPT_DIR=${DA3_CKPT_DIR:-/data/LFT-W02_data/junjie/VLA_WM/WSA/checkpoints/DA3-LARGE-1.1}
DA3_CODE_ROOT=${DA3_CODE_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/Geometric-Action-Model/Depth-Anything-3}
USE_DEPTH=${USE_DEPTH:-1}                 # 1: also align DA3 depth features (aux head)
DA3_PROCESS_RES=${DA3_PROCESS_RES:-224}   # 224 -> 16x16=256 tok (matches grid 16); 504 -> 36x36

DATASET_ROOT=${DATASET_ROOT:?set DATASET_ROOT to the semantic-plan dataset dir}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl_semantic_planner/qwen3vl2b_dinov3_da3}

# plan geometry: 5 keyframes x 16^2=256 tokens; DINOv3 dim 1280, DA3 dim 2048 (auto from teachers)
NUM_KEYFRAMES=${NUM_KEYFRAMES:-5}
GRID_SIZE=${GRID_SIZE:-16}
NUM_LATENT_PER_KEYFRAME=${NUM_LATENT_PER_KEYFRAME:-8}
NUM_HEAD_LATENT_PER_KEYFRAME=${NUM_HEAD_LATENT_PER_KEYFRAME:-0}
SEMANTIC_DIM=${SEMANTIC_DIM:-0}          # 0 -> auto from DINOv3 (1280)
DEPTH_GRID_SIZE=${DEPTH_GRID_SIZE:-16}   # must equal DA3_PROCESS_RES/14
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-49}
KEYFRAME_SCHEME=${KEYFRAME_SCHEME:-uniform}
KEYFRAME_GAMMA=${KEYFRAME_GAMMA:-0.6}
KEYFRAME_OFFSETS=${KEYFRAME_OFFSETS:-}
DINO_INPUT_SIZE=${DINO_INPUT_SIZE:-256}  # DINOv3 input (patch16 -> 256/16=16 grid)

# plain MSE (matches the lingbot line's active video term); other terms off
MSE_LOSS_WEIGHT=${MSE_LOSS_WEIGHT:-1.0}
COSINE_LOSS_WEIGHT=${COSINE_LOSS_WEIGHT:-0.0}
NORM_LOSS_WEIGHT=${NORM_LOSS_WEIGHT:-0.0}
VARIANCE_LOSS_WEIGHT=${VARIANCE_LOSS_WEIGHT:-0.0}
INFONCE_LOSS_WEIGHT=${INFONCE_LOSS_WEIGHT:-0.0}
DEPTH_LOSS_WEIGHT=${DEPTH_LOSS_WEIGHT:-0.004}

BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-1e-5}
HEAD_LR=${HEAD_LR:-1e-4}
MAX_STEPS=${MAX_STEPS:-16000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOG_STEPS=${LOG_STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-400}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
NUM_WORKERS=${NUM_WORKERS:-4}
DTYPE=${DTYPE:-bf16}
FULL_FINETUNE=${FULL_FINETUNE:-1}
FREEZE_VISION=${FREEZE_VISION:-1}

mkdir -p "$OUTPUT_DIR" logs
echo "[launch] 2B dinov3+da3 gpus=$NUM_GPUS model=$MODEL_PATH dataset=$DATASET_ROOT out=$OUTPUT_DIR"

TRAIN_ARGS=(
  --model-path "$MODEL_PATH"
  --dataset-root "$DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --num-keyframes "$NUM_KEYFRAMES"
  --grid-size "$GRID_SIZE"
  --num-latent-per-keyframe "$NUM_LATENT_PER_KEYFRAME"
  --num-head-latent-per-keyframe "$NUM_HEAD_LATENT_PER_KEYFRAME"
  --lr "$LR" --head-lr "$HEAD_LR"
  --plan-head-type lingbot_dino
  --lora-r 0
  --video-target-type dinov3
  --dinov3-model-dir "$DINOV3_MODEL_DIR"
  --semantic-dim "$SEMANTIC_DIM"
  --mse-loss-weight "$MSE_LOSS_WEIGHT"
  --cosine-loss-weight "$COSINE_LOSS_WEIGHT"
  --norm-loss-weight "$NORM_LOSS_WEIGHT"
  --variance-loss-weight "$VARIANCE_LOSS_WEIGHT"
  --infonce-loss-weight "$INFONCE_LOSS_WEIGHT"
  --weight-decay "$WEIGHT_DECAY"
  --warmup-steps "$WARMUP_STEPS"
  --dtype "$DTYPE"
  --save-steps "$SAVE_STEPS"
  --log-steps "$LOG_STEPS"
  --num-workers "$NUM_WORKERS"
  --train-plan-token-embedding
  --ddp-find-unused-parameters
  --online-plan-labels
  --sequence-length "$SEQUENCE_LENGTH"
  --keyframe-scheme "$KEYFRAME_SCHEME"
  --keyframe-gamma "$KEYFRAME_GAMMA"
  --dino-input-size "$DINO_INPUT_SIZE"
)
[[ "$FULL_FINETUNE" == "1" ]] && TRAIN_ARGS+=(--full-finetune)
[[ "$FREEZE_VISION" == "1" ]] && TRAIN_ARGS+=(--freeze-vision) || TRAIN_ARGS+=(--no-freeze-vision)
[[ -n "$KEYFRAME_OFFSETS" ]] && TRAIN_ARGS+=(--keyframe-offsets "$KEYFRAME_OFFSETS")
[[ "$USE_DEPTH" == "1" ]] && TRAIN_ARGS+=(
  --use-depth
  --depth-target-type da3
  --da3-ckpt-dir "$DA3_CKPT_DIR"
  --da3-process-res "$DA3_PROCESS_RES"
  --depth-grid-size "$DEPTH_GRID_SIZE"
  --depth-loss-weight "$DEPTH_LOSS_WEIGHT"
)

TRAIN_SCRIPT="$PLANNER_DIR/train_qwen3vl4b_lingbot_dino_planner.py"
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1
export DINOV3_MODEL_DIR DA3_CKPT_DIR DA3_CODE_ROOT
if [[ "$NUM_GPUS" -gt 1 ]]; then
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
else
  "$PY" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
fi
