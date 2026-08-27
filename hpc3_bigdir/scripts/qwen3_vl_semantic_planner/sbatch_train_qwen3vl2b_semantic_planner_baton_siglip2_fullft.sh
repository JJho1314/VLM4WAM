#!/usr/bin/env bash
# Baton Stage-1: full fine-tune Qwen3-VL planner against frozen SigLIP2 semantic blueprints.

#SBATCH --job-name=q3vl2b-baton-fullft
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-baton-fullft-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-baton-fullft-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT" || exit 2
mkdir -p logs

module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
CONDA_ENV=${CONDA_ENV:-starVLA}
conda activate "$CONDA_ENV" 2>/dev/null || true
PY=${PY:-/data/user/jhe724/.conda/envs/starVLA/bin/python}

MODEL_PATH=${MODEL_PATH:-/data/user/jhe724/workspace/weights/Qwen3-VL-2B-Instruct}
DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
PLAN_LABEL_DIR=${PLAN_LABEL_DIR:-$DATASET_ROOT/siglip2_semantic_plan_k6_g9_full}
OUTPUT_DIR=${OUTPUT_DIR:-/data/user/jhe724/workspace/VLM4WAM/outputs/qwen3vl_semantic_planner/qwen3vl2b_baton_siglip2_k6_g9_fullft_8gpu_b1_acc8_gbs64_3000step}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MAX_STEPS=${MAX_STEPS:-3000}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
NUM_GPUS=${NUM_GPUS:-8}
SAMPLE_ONE_WINDOW_PER_STEM=${SAMPLE_ONE_WINDOW_PER_STEM:-0}
NUM_KEYFRAMES=${NUM_KEYFRAMES:-6}
GRID_SIZE=${GRID_SIZE:-9}
LR=${LR:-2e-6}
HEAD_LR=${HEAD_LR:-1e-4}
PLAN_HEAD_TYPE=${PLAN_HEAD_TYPE:-mlp}
# CoVT bottleneck (plan-head-type=covt): LM emits NUM_LATENT_PER_KEYFRAME latents/keyframe;
# a decoder reconstructs the dense GRID_SIZE x GRID_SIZE SigLIP grid. Ignored by mlp/baton.
NUM_LATENT_PER_KEYFRAME=${NUM_LATENT_PER_KEYFRAME:-4}
# Online labels (cosmos-style episode sampling): random window per stem per epoch from
# frame_ranges.json, SigLIP2 targets encoded on the fly. PLAN_LABEL_DIR is unused then.
ONLINE_PLAN_LABELS=${ONLINE_PLAN_LABELS:-0}
SIGLIP2_ENCODER_PATH=${SIGLIP2_ENCODER_PATH:-/data/user/jhe724/workspace/weights/siglip2-so400m-patch14-384}
KEYFRAME_SCHEME=${KEYFRAME_SCHEME:-uniform}
KEYFRAME_GAMMA=${KEYFRAME_GAMMA:-0.6}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-49}
ONLINE_GRID_SIZE=${ONLINE_GRID_SIZE:-0}
SEMANTIC_DIM=${SEMANTIC_DIM:-0}
PLAN_HEAD_NUM_HEADS=${PLAN_HEAD_NUM_HEADS:-16}
PLAN_HEAD_DROPOUT=${PLAN_HEAD_DROPOUT:-0.0}
SEM_MLP_HIDDEN_SIZE=${SEM_MLP_HIDDEN_SIZE:--1}
# Anti-collapse loss recipe (matches the train script defaults). Plain MSE alone regresses
# multimodal futures to their per-sample mean; cosine/norm/variance/infonce keep the plan
# discriminative. Do NOT set COSINE_LOSS_WEIGHT=0 — that was the pre-fix recipe.
MSE_LOSS_WEIGHT=${MSE_LOSS_WEIGHT:-1.0}
COSINE_LOSS_WEIGHT=${COSINE_LOSS_WEIGHT:-1.0}
NORM_LOSS_WEIGHT=${NORM_LOSS_WEIGHT:-0.2}
VARIANCE_LOSS_WEIGHT=${VARIANCE_LOSS_WEIGHT:-0.1}
INFONCE_LOSS_WEIGHT=${INFONCE_LOSS_WEIGHT:-0.1}
INFONCE_TEMPERATURE=${INFONCE_TEMPERATURE:-0.07}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_STEPS=${WARMUP_STEPS:-100}
LR_SCHEDULE=${LR_SCHEDULE:-cosine}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
DTYPE=${DTYPE:-bf16}
SAVE_STEPS=${SAVE_STEPS:-500}
NUM_WORKERS=${NUM_WORKERS:-2}
FREEZE_VISION=${FREEZE_VISION:-1}
FREEZE_LM_HEAD=${FREEZE_LM_HEAD:-1}

# HPC3 compute nodes have no internet: offline wandb only (sync later with `wandb sync`).
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-qwen3vl_semantic_planner}

export PLAN_LABEL_DIR

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: Qwen3-VL-2B model path not found: $MODEL_PATH" >&2
  exit 2
fi
if [[ "$ONLINE_PLAN_LABELS" != "1" && ! -d "$PLAN_LABEL_DIR" ]]; then
  echo "ERROR: semantic plan label dir not found: $PLAN_LABEL_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
echo "mode=Qwen3-VL Baton Stage-1 full fine-tune"
echo "model=$MODEL_PATH"
echo "dataset=$DATASET_ROOT"
echo "labels=$PLAN_LABEL_DIR"
echo "output=$OUTPUT_DIR"
echo "max_samples=$MAX_SAMPLES max_steps=$MAX_STEPS batch_per_gpu=$BATCH_SIZE accum=$GRAD_ACCUM num_gpus=$NUM_GPUS global_batch=$((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "sample_one_window_per_stem=$SAMPLE_ONE_WINDOW_PER_STEM"
echo "keyframes=$NUM_KEYFRAMES grid=$GRID_SIZE lr=$LR head_lr=$HEAD_LR plan_head_type=$PLAN_HEAD_TYPE num_latent_per_keyframe=$NUM_LATENT_PER_KEYFRAME plan_head_num_heads=$PLAN_HEAD_NUM_HEADS plan_head_dropout=$PLAN_HEAD_DROPOUT sem_mlp_hidden_size=$SEM_MLP_HIDDEN_SIZE"
echo "loss: mse=$MSE_LOSS_WEIGHT cosine=$COSINE_LOSS_WEIGHT norm=$NORM_LOSS_WEIGHT variance=$VARIANCE_LOSS_WEIGHT infonce=$INFONCE_LOSS_WEIGHT temp=$INFONCE_TEMPERATURE | sched=$LR_SCHEDULE warmup=$WARMUP_STEPS min_lr_ratio=$MIN_LR_RATIO weight_decay=$WEIGHT_DECAY"
echo "freeze_vision=$FREEZE_VISION freeze_lm_head=$FREEZE_LM_HEAD"
echo "NOTE: num-keyframes/grid-size MUST match the world model's SEMANTIC_PLAN_NUM_KEYFRAMES/SPATIAL_GRID for the plan to be consumable."

if [[ "$ONLINE_PLAN_LABELS" != "1" ]]; then
"$PY" - <<'PY'
import json
import pathlib
import torch

label_dir = pathlib.Path(__import__("os").environ["PLAN_LABEL_DIR"])
paths = sorted(label_dir.glob("*.pt"))
manifest = label_dir / "manifest.jsonl"
print(json.dumps({"label_dir": str(label_dir), "pt_count": len(paths), "manifest": manifest.exists()}), flush=True)
if not paths:
    raise SystemExit("No semantic plan label files found")
payload = torch.load(paths[0], map_location="cpu", weights_only=False)
print(json.dumps({
    "sample": paths[0].name,
    "semantic_plan_shape": list(payload["semantic_plan"].shape),
    "feature_type": payload.get("feature_type", "unknown"),
    "prompt": payload.get("prompt", ""),
    "video_path": payload.get("video_path", ""),
    "first_frame_index": payload.get("first_frame_index", 0),
}), flush=True)
PY
else
  echo "online_plan_labels=1 keyframe_scheme=$KEYFRAME_SCHEME gamma=$KEYFRAME_GAMMA seq_len=$SEQUENCE_LENGTH online_grid=$ONLINE_GRID_SIZE siglip2=$SIGLIP2_ENCODER_PATH"
fi

TRAIN_ARGS=(
  --model-path "$MODEL_PATH"
  --dataset-root "$DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --max-samples "$MAX_SAMPLES"
  --max-steps "$MAX_STEPS"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --num-keyframes "$NUM_KEYFRAMES"
  --grid-size "$GRID_SIZE"
  --lr "$LR"
  --head-lr "$HEAD_LR"
  --plan-head-type "$PLAN_HEAD_TYPE"
  --num-latent-per-keyframe "$NUM_LATENT_PER_KEYFRAME"
  --plan-head-num-heads "$PLAN_HEAD_NUM_HEADS"
  --plan-head-dropout "$PLAN_HEAD_DROPOUT"
  --lora-r 0
  --sem-mlp-hidden-size "$SEM_MLP_HIDDEN_SIZE"
  --mse-loss-weight "$MSE_LOSS_WEIGHT"
  --cosine-loss-weight "$COSINE_LOSS_WEIGHT"
  --norm-loss-weight "$NORM_LOSS_WEIGHT"
  --variance-loss-weight "$VARIANCE_LOSS_WEIGHT"
  --infonce-loss-weight "$INFONCE_LOSS_WEIGHT"
  --infonce-temperature "$INFONCE_TEMPERATURE"
  --weight-decay "$WEIGHT_DECAY"
  --warmup-steps "$WARMUP_STEPS"
  --lr-schedule "$LR_SCHEDULE"
  --min-lr-ratio "$MIN_LR_RATIO"
  --dtype "$DTYPE"
  --save-steps "$SAVE_STEPS"
  --num-workers "$NUM_WORKERS"
  --full-finetune
  --ddp-find-unused-parameters
  --train-plan-token-embedding
)

if [[ "$ONLINE_PLAN_LABELS" == "1" ]]; then
  TRAIN_ARGS+=(
    --online-plan-labels
    --siglip2-encoder-path "$SIGLIP2_ENCODER_PATH"
    --sequence-length "$SEQUENCE_LENGTH"
    --keyframe-scheme "$KEYFRAME_SCHEME"
    --keyframe-gamma "$KEYFRAME_GAMMA"
    --online-grid-size "$ONLINE_GRID_SIZE"
    --semantic-dim "$SEMANTIC_DIM"
  )
else
  TRAIN_ARGS+=(--plan-label-dir "$PLAN_LABEL_DIR")
fi

if [[ "$SAMPLE_ONE_WINDOW_PER_STEM" == "1" ]]; then
  TRAIN_ARGS+=(--sample-one-window-per-stem)
fi

if [[ "$FREEZE_VISION" == "1" ]]; then
  TRAIN_ARGS+=(--freeze-vision)
else
  TRAIN_ARGS+=(--no-freeze-vision)
fi

if [[ "$FREEZE_LM_HEAD" == "1" ]]; then
  TRAIN_ARGS+=(--freeze-lm-head)
else
  TRAIN_ARGS+=(--no-freeze-lm-head)
fi

# Overridable entry script (backward-compatible: defaults to the SigLIP CoVT planner). Independent
# variants (e.g. the tasktoken planner) set TRAIN_SCRIPT to their own file without touching this harness.
TRAIN_SCRIPT=${TRAIN_SCRIPT:-scripts/qwen3_vl_semantic_planner/train_qwen3vl_semantic_planner.py}
if [[ "$NUM_GPUS" -gt 1 ]]; then
  "$PY" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    "$TRAIN_SCRIPT" \
    "${TRAIN_ARGS[@]}"
else
  "$PY" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
fi
