#!/bin/bash
# Self-contained launcher for the 4B lingbot-DINO planner line (independent of the 2B slurm base).
# Runs directly via torchrun in the box env we validated (torch 2.8+cu128, transformers 5.5.4,
# flex_attention). All paths default to the downloaded/extracted weights on this box.
#
# Required:  DATASET_ROOT  (the DROID semantic-plan dataset — must exist wherever you run this)
# Smoke:     NUM_GPUS=1 BATCH_SIZE=1 GRAD_ACCUM=1 MAX_STEPS=2 FULL_FINETUNE=0 bash train_lingbot_dino_4b.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANNER_DIR="$(dirname "$HERE")"                       # scripts/qwen3_vl_semantic_planner
REPO_ROOT="$(cd "$PLANNER_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

# covt/planner training env: transformers 4.57 (Qwen3-VL) + torch 2.6 flex_attention + mistral_common 1.9
# (the box default python has transformers 5.x with a broken mistral_common, so pin this env).
PY=${PY:-/data/LFT-W02_data/.conda/envs/starVLA/bin/python}
NUM_GPUS=${NUM_GPUS:-2}

WEIGHTS=${WEIGHTS:-/data/LFT-W02_data/junjie/weights}
MODEL_PATH=${MODEL_PATH:-$WEIGHTS/Qwen3-VL-4B-lingbot-vlm}          # extracted 4B VLM (see extract_qwenvl_from_lingbot.py)
LINGBOT_6B=${LINGBOT_6B:-$WEIGHTS/lingbot-vla-v2-6b}
DINO_TEACHER_CKPT=${DINO_TEACHER_CKPT:-$LINGBOT_6B/dino_video/teacher_step_10000.pth}
DINO_TEACHER_CONFIG=${DINO_TEACHER_CONFIG:-$LINGBOT_6B/dino_video/config.yaml}
HEAD_WARMSTART_CKPT=${HEAD_WARMSTART_CKPT:-$LINGBOT_6B}             # warm-start head from future_video_align_head.*

# --- auxiliary future-DEPTH alignment (lingbot-style: MoGe-2 -> MoRGBD, smooth_L1; OFF by default) ---
USE_DEPTH=${USE_DEPTH:-0}
DEPTH_MOGE_PATH=${DEPTH_MOGE_PATH:-$WEIGHTS/moge-2-vitb-normal/model.pt}   # MoGe-2 (HF Ruicheng/moge-2-vitb-normal)
DEPTH_MORGBD_PATH=${DEPTH_MORGBD_PATH:-$LINGBOT_6B/depth/model.pt}         # LingBot-Depth / MoRGBD
DEPTH_LOSS_WEIGHT=${DEPTH_LOSS_WEIGHT:-0.004}   # lingbot future_depth_loss_weight
DEPTH_GRID_SIZE=${DEPTH_GRID_SIZE:-16}          # 16 -> 256 depth tokens/keyframe
DEPTH_DIM=${DEPTH_DIM:-1024}

# lingbot source (moge/mdm/dino teacher code) + MoGe-pinned utils3d; override on non-box hosts (HPC3).
LINGBOT_SRC_ROOT=${LINGBOT_SRC_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/lingbot-vla-v2}
UTILS3D_MOGE_PATH=${UTILS3D_MOGE_PATH:-/data/LFT-W02_data/junjie/weights/py_deps/utils3d_moge}

DATASET_ROOT=${DATASET_ROOT:?set DATASET_ROOT to the DROID semantic-plan dataset dir}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl_semantic_planner/qwen3vl4b_lingbot_dino_uniform_k5}

# plan geometry: DINO-video, 5 keyframes x 16^2=256 tokens x 1024 dim => target_len 1280
NUM_KEYFRAMES=${NUM_KEYFRAMES:-5}
GRID_SIZE=${GRID_SIZE:-16}
NUM_LATENT_PER_KEYFRAME=${NUM_LATENT_PER_KEYFRAME:-8}   # matches lingbot num_task_tokens=8
SEMANTIC_DIM=${SEMANTIC_DIM:-1024}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-49}
KEYFRAME_SCHEME=${KEYFRAME_SCHEME:-uniform}
KEYFRAME_GAMMA=${KEYFRAME_GAMMA:-0.6}
DINO_INPUT_SIZE=${DINO_INPUT_SIZE:-256}

# plain MSE (lingbot's active video term); other terms off
MSE_LOSS_WEIGHT=${MSE_LOSS_WEIGHT:-1.0}
COSINE_LOSS_WEIGHT=${COSINE_LOSS_WEIGHT:-0.0}
NORM_LOSS_WEIGHT=${NORM_LOSS_WEIGHT:-0.0}
VARIANCE_LOSS_WEIGHT=${VARIANCE_LOSS_WEIGHT:-0.0}
INFONCE_LOSS_WEIGHT=${INFONCE_LOSS_WEIGHT:-0.0}

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
FULL_FINETUNE=${FULL_FINETUNE:-1}   # 1: tune LM+head; 0: head + plan-token embeddings only (fits small GPUs)

mkdir -p "$OUTPUT_DIR" logs
echo "[launch] gpus=$NUM_GPUS model=$MODEL_PATH dataset=$DATASET_ROOT out=$OUTPUT_DIR full_ft=$FULL_FINETUNE"

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
  --lr "$LR"
  --head-lr "$HEAD_LR"
  --plan-head-type lingbot_dino
  --lora-r 0
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
  --freeze-vision
  --train-plan-token-embedding
  --ddp-find-unused-parameters
  --online-plan-labels
  --sequence-length "$SEQUENCE_LENGTH"
  --keyframe-scheme "$KEYFRAME_SCHEME"
  --keyframe-gamma "$KEYFRAME_GAMMA"
  --semantic-dim "$SEMANTIC_DIM"
  --dino-teacher-ckpt "$DINO_TEACHER_CKPT"
  --dino-teacher-config "$DINO_TEACHER_CONFIG"
  --dino-input-size "$DINO_INPUT_SIZE"
)
# warm-start only if the 6b model shards are actually present (skip gracefully, e.g. smoke on HPC3).
if [[ -n "$HEAD_WARMSTART_CKPT" && -f "$HEAD_WARMSTART_CKPT/model.safetensors.index.json" ]]; then
  TRAIN_ARGS+=(--head-warmstart-ckpt "$HEAD_WARMSTART_CKPT")
else
  echo "[launch] skip head warm-start (no model.safetensors.index.json under '$HEAD_WARMSTART_CKPT')"
fi
[[ "$FULL_FINETUNE" == "1" ]] && TRAIN_ARGS+=(--full-finetune)
[[ "$USE_DEPTH" == "1" ]] && TRAIN_ARGS+=(
  --use-depth
  --depth-moge-path "$DEPTH_MOGE_PATH"
  --depth-morgbd-path "$DEPTH_MORGBD_PATH"
  --depth-loss-weight "$DEPTH_LOSS_WEIGHT"
  --depth-grid-size "$DEPTH_GRID_SIZE"
  --depth-dim "$DEPTH_DIM"
)

TRAIN_SCRIPT="$PLANNER_DIR/train_qwen3vl4b_lingbot_dino_planner.py"
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1   # MoGe/MoRGBD dinov2 -> F.scaled_dot_product_attention (unbuilt xformers op)
export LINGBOT_SRC_ROOT UTILS3D_MOGE_PATH
if [[ "$NUM_GPUS" -gt 1 ]]; then
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
else
  "$PY" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
fi
