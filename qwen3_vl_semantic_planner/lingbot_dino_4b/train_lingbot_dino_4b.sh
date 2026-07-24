#!/bin/bash
# Self-contained launcher for the 4B lingbot-DINO planner line (independent of the 2B slurm base).
# Runs through Accelerate + DeepSpeed ZeRO-2 by default in the validated box env.
# All paths default to the downloaded/extracted weights on this box.
#
# Required: exactly one of DATASET_ROOT, FASTWAM_DATA_CONFIG, or GE_ACT_DATA_CONFIG.
# Smoke:     USE_DEEPSPEED=0 NUM_GPUS=1 BATCH_SIZE=1 GRAD_ACCUM=1 EXPECTED_GLOBAL_BATCH=1 MAX_STEPS=2 SAVE_STEPS=2 FULL_FINETUNE=0 bash train_lingbot_dino_4b.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLANNER_DIR="$(dirname "$HERE")"                       # qwen3_vl_semantic_planner
REPO_ROOT="$(cd "$PLANNER_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 2

# covt/planner training env: transformers 4.57 (Qwen3-VL) + torch 2.6 flex_attention + mistral_common 1.9
# (the box default python has transformers 5.x with a broken mistral_common, so pin this env).
PY=${PY:-/data/LFT-W02_data/.conda/envs/starVLA/bin/python}
NUM_GPUS=${NUM_GPUS:-8}
USE_DEEPSPEED=${USE_DEEPSPEED:-1}

WEIGHTS=${WEIGHTS:-/data/LFT-W02_data/junjie/weights}
MODEL_PATH=${MODEL_PATH:-$WEIGHTS/Qwen3-VL-4B-lingbot-vlm}          # extracted 4B VLM (see extract_qwenvl_from_lingbot.py)
INIT_PLANNER_CHECKPOINT=${INIT_PLANNER_CHECKPOINT-}
LINGBOT_6B=${LINGBOT_6B:-$WEIGHTS/lingbot-vla-v2-6b}
DINO_TEACHER_CKPT=${DINO_TEACHER_CKPT:-$LINGBOT_6B/dino_video/teacher_step_10000.pth}
DINO_TEACHER_CONFIG=${DINO_TEACHER_CONFIG:-$LINGBOT_6B/dino_video/config.yaml}
HEAD_WARMSTART_CKPT=${HEAD_WARMSTART_CKPT-$LINGBOT_6B}             # warm-start head from future_video_align_head.*

# --- auxiliary future-DEPTH alignment (lingbot-style: MoGe-2 -> MoRGBD, smooth_L1) ---
USE_DEPTH=${USE_DEPTH:-1}
DEPTH_MOGE_PATH=${DEPTH_MOGE_PATH:-$WEIGHTS/moge-2-vitb-normal/model.pt}   # MoGe-2 (HF Ruicheng/moge-2-vitb-normal)
DEPTH_MORGBD_PATH=${DEPTH_MORGBD_PATH:-$LINGBOT_6B/depth/model.pt}         # LingBot-Depth / MoRGBD
DEPTH_LOSS_WEIGHT=${DEPTH_LOSS_WEIGHT:-0.004}   # lingbot future_depth_loss_weight
DEPTH_GRID_SIZE=${DEPTH_GRID_SIZE:-16}          # 16 -> 256 depth tokens/keyframe
DEPTH_DIM=${DEPTH_DIM:-1024}
# --- teacher swap for the 2B DINOv3+DA3 line (opt-in; default = lingbot dino_video/morgbd) ---
VIDEO_TARGET_TYPE=${VIDEO_TARGET_TYPE:-dino_video}
DEPTH_TARGET_TYPE=${DEPTH_TARGET_TYPE:-morgbd}
DINOV3_MODEL_DIR=${DINOV3_MODEL_DIR:-}
SIGLIP2_MODEL_DIR=${SIGLIP2_MODEL_DIR:-}
SIGLIP2_GRID_SIZE=${SIGLIP2_GRID_SIZE:-16}   # pool SigLIP2's penultimate grid -> 256 tok (DA3-matched)
SIGLIP2_INPUT_SIZE=${SIGLIP2_INPUT_SIZE:-384}  # 384 = native (label-exact); 224 = 16x16 native tok, ~2.8x faster
DA3_CKPT_DIR=${DA3_CKPT_DIR:-}
DA3_PROCESS_RES=${DA3_PROCESS_RES:-224}
DA3_ALIGN_STRATEGY=${DA3_ALIGN_STRATEGY:-last_layer}     # last_layer (default) | wsa_multilayer
DA3_TEACHER_LAYERS=${DA3_TEACHER_LAYERS:-11,15,19,23}     # wsa_multilayer: abs backbone layers
DA3_LAYER_WEIGHTS=${DA3_LAYER_WEIGHTS:-1.0,1.2,1.4,1.6}   # wsa_multilayer: per-layer loss weights
USE_CURRENT_ALIGNMENT=${USE_CURRENT_ALIGNMENT:-1}
BIDIRECTIONAL_PLAN_ATTN=${BIDIRECTIONAL_PLAN_ATTN:-0}   # 1: official-parity bidirectional prefix attention (needs retrain)
INDEPENDENT_MODALITY_TASK_TOKENS=${INDEPENDENT_MODALITY_TASK_TOKENS:-1}
NUM_TASK_TOKENS=${NUM_TASK_TOKENS:-64}
CURRENT_DINO_LOSS_WEIGHT=${CURRENT_DINO_LOSS_WEIGHT:-0.004}
FUTURE_DINO_LOSS_WEIGHT=${FUTURE_DINO_LOSS_WEIGHT:-0.004}
CURRENT_DEPTH_LOSS_WEIGHT=${CURRENT_DEPTH_LOSS_WEIGHT:-0.004}
FUTURE_DEPTH_LOSS_WEIGHT=${FUTURE_DEPTH_LOSS_WEIGHT:-0.004}

# lingbot source (moge/mdm/dino teacher code) + MoGe-pinned utils3d; override on non-box hosts (HPC3).
LINGBOT_SRC_ROOT=${LINGBOT_SRC_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/lingbot-vla-v2}
UTILS3D_MOGE_PATH=${UTILS3D_MOGE_PATH:-/data/LFT-W02_data/junjie/weights/py_deps/utils3d_moge}

DATASET_ROOT=${DATASET_ROOT:-}
FASTWAM_DATA_CONFIG=${FASTWAM_DATA_CONFIG:-}
GE_ACT_DATA_CONFIG=${GE_ACT_DATA_CONFIG:-}
FASTWAM_DATASET_DIRS=${FASTWAM_DATASET_DIRS:-}
FASTWAM_TEXT_EMBEDDING_CACHE_DIR=${FASTWAM_TEXT_EMBEDDING_CACHE_DIR:-}
FASTWAM_PRETRAINED_NORM_STATS=${FASTWAM_PRETRAINED_NORM_STATS:-}
DATA_SOURCE_COUNT=0
[[ -n "$DATASET_ROOT" ]] && DATA_SOURCE_COUNT=$((DATA_SOURCE_COUNT + 1))
[[ -n "$FASTWAM_DATA_CONFIG" ]] && DATA_SOURCE_COUNT=$((DATA_SOURCE_COUNT + 1))
[[ -n "$GE_ACT_DATA_CONFIG" ]] && DATA_SOURCE_COUNT=$((DATA_SOURCE_COUNT + 1))
if [[ "$DATA_SOURCE_COUNT" -ne 1 ]]; then
  echo "[launch] set exactly one of DATASET_ROOT, FASTWAM_DATA_CONFIG, or GE_ACT_DATA_CONFIG" >&2
  exit 2
fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl_semantic_planner/qwen3vl4b_lingbot_dino_uniform_k5}

# plan geometry: one future keyframe at offset 8, 16^2=256 tokens x 1024 dim
NUM_KEYFRAMES=${NUM_KEYFRAMES:-1}
GRID_SIZE=${GRID_SIZE:-16}
NUM_LATENT_PER_KEYFRAME=${NUM_LATENT_PER_KEYFRAME:-8}   # legacy fallback; independent q64 uses NUM_TASK_TOKENS
SHARED_LATENT_PER_KEYFRAME=${SHARED_LATENT_PER_KEYFRAME:-32}
PRIVATE_LATENT_PER_KEYFRAME=${PRIVATE_LATENT_PER_KEYFRAME:-32}
SEMANTIC_DIM=${SEMANTIC_DIM:-1024}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-9}
KEYFRAME_SCHEME=${KEYFRAME_SCHEME:-even_future}
KEYFRAME_GAMMA=${KEYFRAME_GAMMA:-0.6}
DINO_INPUT_SIZE=${DINO_INPUT_SIZE:-256}

# plain MSE (lingbot's active video term); other terms off
MSE_LOSS_WEIGHT=${MSE_LOSS_WEIGHT:-1.0}
COSINE_LOSS_WEIGHT=${COSINE_LOSS_WEIGHT:-0.0}
NORM_LOSS_WEIGHT=${NORM_LOSS_WEIGHT:-0.0}
VARIANCE_LOSS_WEIGHT=${VARIANCE_LOSS_WEIGHT:-0.0}
INFONCE_LOSS_WEIGHT=${INFONCE_LOSS_WEIGHT:-0.0}

BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
EXPECTED_GLOBAL_BATCH=${EXPECTED_GLOBAL_BATCH:-128}
LR=${LR:-3e-5}
HEAD_LR=${HEAD_LR:-3e-4}
MAX_STEPS=${MAX_STEPS:-12000}
SAVE_STEPS=${SAVE_STEPS:-1000}
SAVE_START_STEP=${SAVE_START_STEP:-0}
LOG_STEPS=${LOG_STEPS:-10}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
NUM_WORKERS=${NUM_WORKERS:-4}
DTYPE=${DTYPE:-bf16}
FULL_FINETUNE=${FULL_FINETUNE:-1}   # 1: tune LM+head; 0: head + plan-token embeddings only (fits small GPUs)

mkdir -p "$OUTPUT_DIR" logs
DATA_SOURCE=${GE_ACT_DATA_CONFIG:-${FASTWAM_DATA_CONFIG:-$DATASET_ROOT}}
echo "[launch] gpus=$NUM_GPUS model=$MODEL_PATH dataset=$DATA_SOURCE out=$OUTPUT_DIR full_ft=$FULL_FINETUNE"

TRAIN_ARGS=(
  --output-dir "$OUTPUT_DIR"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --expected-global-batch "$EXPECTED_GLOBAL_BATCH"
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
  --save-start-step "$SAVE_START_STEP"
  --log-steps "$LOG_STEPS"
  --num-workers "$NUM_WORKERS"
  --freeze-vision
  --train-plan-token-embedding
  --online-plan-labels
  --sequence-length "$SEQUENCE_LENGTH"
  --keyframe-scheme "$KEYFRAME_SCHEME"
  --keyframe-gamma "$KEYFRAME_GAMMA"
  --semantic-dim "$SEMANTIC_DIM"
  --dino-teacher-ckpt "$DINO_TEACHER_CKPT"
  --dino-teacher-config "$DINO_TEACHER_CONFIG"
  --dino-input-size "$DINO_INPUT_SIZE"
  --video-target-type "$VIDEO_TARGET_TYPE"
)
if [[ -n "$INIT_PLANNER_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--init-planner-checkpoint "$INIT_PLANNER_CHECKPOINT")
else
  TRAIN_ARGS+=(--model-path "$MODEL_PATH")
fi
if [[ -n "$GE_ACT_DATA_CONFIG" ]]; then
  TRAIN_ARGS+=(--ge-act-data-config "$GE_ACT_DATA_CONFIG")
elif [[ -n "$FASTWAM_DATA_CONFIG" ]]; then
  export PYTHONPATH="$REPO_ROOT/third_party/FastWAM/src${PYTHONPATH:+:$PYTHONPATH}"
  TRAIN_ARGS+=(--fastwam-data-config "$FASTWAM_DATA_CONFIG")
  if [[ -n "$FASTWAM_DATASET_DIRS" ]]; then
    IFS=':' read -r -a fastwam_dataset_dirs <<< "$FASTWAM_DATASET_DIRS"
    for dataset_dir in "${fastwam_dataset_dirs[@]}"; do
      [[ -n "$dataset_dir" ]] && TRAIN_ARGS+=(--fastwam-dataset-dir "$dataset_dir")
    done
  fi
  if [[ -n "$FASTWAM_TEXT_EMBEDDING_CACHE_DIR" ]]; then
    TRAIN_ARGS+=(
      --fastwam-text-embedding-cache-dir "$FASTWAM_TEXT_EMBEDDING_CACHE_DIR"
    )
  fi
  if [[ -n "$FASTWAM_PRETRAINED_NORM_STATS" ]]; then
    TRAIN_ARGS+=(
      --fastwam-pretrained-norm-stats "$FASTWAM_PRETRAINED_NORM_STATS"
    )
  fi
else
  TRAIN_ARGS+=(--dataset-root "$DATASET_ROOT")
fi
if [[ "$SHARED_LATENT_PER_KEYFRAME" =~ ^[1-9][0-9]*$ ]]; then
  TRAIN_ARGS+=(--shared-latent-per-keyframe "$SHARED_LATENT_PER_KEYFRAME")
fi
if [[ "$PRIVATE_LATENT_PER_KEYFRAME" =~ ^[1-9][0-9]*$ ]]; then
  TRAIN_ARGS+=(--private-latent-per-keyframe "$PRIVATE_LATENT_PER_KEYFRAME")
fi
# warm-start only if the 6b model shards are actually present (skip gracefully, e.g. smoke on HPC3).
if [[ -z "$INIT_PLANNER_CHECKPOINT" && -n "$HEAD_WARMSTART_CKPT" && -f "$HEAD_WARMSTART_CKPT/model.safetensors.index.json" ]]; then
  TRAIN_ARGS+=(--head-warmstart-ckpt "$HEAD_WARMSTART_CKPT")
else
  echo "[launch] skip head warm-start (no model.safetensors.index.json under '$HEAD_WARMSTART_CKPT')"
fi
[[ "$FULL_FINETUNE" == "1" ]] && TRAIN_ARGS+=(--full-finetune)
[[ -n "$DINOV3_MODEL_DIR" ]] && TRAIN_ARGS+=(--dinov3-model-dir "$DINOV3_MODEL_DIR")
if [[ "$VIDEO_TARGET_TYPE" == "siglip2" ]]; then
  TRAIN_ARGS+=(--siglip2-grid-size "$SIGLIP2_GRID_SIZE" --siglip2-input-size "$SIGLIP2_INPUT_SIZE")
  [[ -n "$SIGLIP2_MODEL_DIR" ]] && TRAIN_ARGS+=(--siglip2-model-dir "$SIGLIP2_MODEL_DIR")
fi
if [[ "$USE_DEPTH" == "1" ]]; then
  TRAIN_ARGS+=(--use-depth --depth-target-type "$DEPTH_TARGET_TYPE" --depth-loss-weight "$DEPTH_LOSS_WEIGHT" --depth-grid-size "$DEPTH_GRID_SIZE")
  if [[ "$DEPTH_TARGET_TYPE" == "da3" ]]; then
    TRAIN_ARGS+=(--da3-process-res "$DA3_PROCESS_RES" --da3-align-strategy "$DA3_ALIGN_STRATEGY")
    [[ -n "$DA3_CKPT_DIR" ]] && TRAIN_ARGS+=(--da3-ckpt-dir "$DA3_CKPT_DIR")
    if [[ "$DA3_ALIGN_STRATEGY" == "wsa_multilayer" ]]; then
      TRAIN_ARGS+=(--da3-teacher-layers "$DA3_TEACHER_LAYERS" --da3-layer-weights "$DA3_LAYER_WEIGHTS")
    fi
  else
    TRAIN_ARGS+=(--depth-moge-path "$DEPTH_MOGE_PATH" --depth-morgbd-path "$DEPTH_MORGBD_PATH" --depth-dim "$DEPTH_DIM")
  fi
fi
[[ "$USE_CURRENT_ALIGNMENT" == "1" ]] && TRAIN_ARGS+=(
  --use-current-alignment
  --num-task-tokens "$NUM_TASK_TOKENS"
  --current-dino-loss-weight "$CURRENT_DINO_LOSS_WEIGHT"
  --future-dino-loss-weight "$FUTURE_DINO_LOSS_WEIGHT"
  --current-depth-loss-weight "$CURRENT_DEPTH_LOSS_WEIGHT"
  --future-depth-loss-weight "$FUTURE_DEPTH_LOSS_WEIGHT"
)
[[ "$INDEPENDENT_MODALITY_TASK_TOKENS" == "1" ]] && TRAIN_ARGS+=(
  --independent-modality-task-tokens
)
[[ "$BIDIRECTIONAL_PLAN_ATTN" == "1" ]] && TRAIN_ARGS+=(--bidirectional-plan-attn)

TRAIN_SCRIPT="$PLANNER_DIR/train_semantic_planner.py"
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1   # MoGe/MoRGBD dinov2 -> F.scaled_dot_product_attention (unbuilt xformers op)
export LINGBOT_SRC_ROOT UTILS3D_MOGE_PATH
CONFIG_DIR="$OUTPUT_DIR/runtime_config"
mkdir -p "$CONFIG_DIR"
if [[ "$USE_DEEPSPEED" == "1" ]]; then
  ACCELERATE_CONFIG=$(
    "$PY" "$HERE/make_zero2_config.py" \
      --grad-accum "$GRAD_ACCUM" \
      --num-processes "$NUM_GPUS" \
      --output-dir "$CONFIG_DIR"
  )
  exec "$PY" -m accelerate.commands.launch \
    --config_file "$ACCELERATE_CONFIG" \
    "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
fi
exec "$PY" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
