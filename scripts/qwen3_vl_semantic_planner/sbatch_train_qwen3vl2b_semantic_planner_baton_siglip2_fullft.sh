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
PLAN_HEAD_NUM_HEADS=${PLAN_HEAD_NUM_HEADS:-16}
PLAN_HEAD_DROPOUT=${PLAN_HEAD_DROPOUT:-0.0}
SEM_MLP_HIDDEN_SIZE=${SEM_MLP_HIDDEN_SIZE:--1}
COSINE_LOSS_WEIGHT=${COSINE_LOSS_WEIGHT:-0.0}
DTYPE=${DTYPE:-bf16}
SAVE_STEPS=${SAVE_STEPS:-500}
NUM_WORKERS=${NUM_WORKERS:-2}
FREEZE_VISION=${FREEZE_VISION:-1}
FREEZE_LM_HEAD=${FREEZE_LM_HEAD:-1}

export PLAN_LABEL_DIR

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: Qwen3-VL-2B model path not found: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$PLAN_LABEL_DIR" ]]; then
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
echo "keyframes=$NUM_KEYFRAMES grid=$GRID_SIZE lr=$LR head_lr=$HEAD_LR plan_head_type=$PLAN_HEAD_TYPE plan_head_num_heads=$PLAN_HEAD_NUM_HEADS plan_head_dropout=$PLAN_HEAD_DROPOUT sem_mlp_hidden_size=$SEM_MLP_HIDDEN_SIZE cosine_loss_weight=$COSINE_LOSS_WEIGHT"
echo "freeze_vision=$FREEZE_VISION freeze_lm_head=$FREEZE_LM_HEAD"

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

TRAIN_ARGS=(
  --model-path "$MODEL_PATH"
  --dataset-root "$DATASET_ROOT"
  --plan-label-dir "$PLAN_LABEL_DIR"
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
  --plan-head-num-heads "$PLAN_HEAD_NUM_HEADS"
  --plan-head-dropout "$PLAN_HEAD_DROPOUT"
  --lora-r 0
  --sem-mlp-hidden-size "$SEM_MLP_HIDDEN_SIZE"
  --cosine-loss-weight "$COSINE_LOSS_WEIGHT"
  --dtype "$DTYPE"
  --save-steps "$SAVE_STEPS"
  --num-workers "$NUM_WORKERS"
  --full-finetune
  --ddp-find-unused-parameters
  --train-plan-token-embedding
)

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

if [[ "$NUM_GPUS" -gt 1 ]]; then
  "$PY" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    scripts/qwen3_vl_semantic_planner/train_qwen3vl_semantic_planner.py \
    "${TRAIN_ARGS[@]}"
else
  "$PY" scripts/qwen3_vl_semantic_planner/train_qwen3vl_semantic_planner.py "${TRAIN_ARGS[@]}"
fi
