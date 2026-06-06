#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"

VENV=${VENV:-/data/LFT-W02_data/junjie/cosmos-predict2.5/.venv}
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONHOME

NV_LIB="$VENV/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-cuda:$REPO_ROOT/packages/cosmos-oss:${PYTHONPATH:-}"

export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/LFT-W02_data/junjie/weights}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://10.12.1.245:8080}
export WANDB_API_KEY=${WANDB_API_KEY:-local-37151658708fac20809135dce9e234842db32f97}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export COSMOS_SKIP_CUDA_VERSION_CHECK=${COSMOS_SKIP_CUDA_VERSION_CHECK:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/outputs/val_one_per_scene_instructsam_feature_iter10000_20260603_015800}
EVAL_DATASET_DIR=${EVAL_DATASET_DIR:-$RUN_ROOT/dataset_one_per_scene}
TEXT_ATTN_OUT=${TEXT_ATTN_OUT:-$RUN_ROOT/text_target_attention}
FEATURE_ATTN_OUT=${FEATURE_ATTN_OUT:-$RUN_ROOT/instructsam_feature_attention}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}
ATTN_CHECKPOINT=${ATTN_CHECKPOINT:-$REPO_ROOT/outputs/pulled_checkpoints/instructsam_iter000010000_dcp}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context}
VIZ_BLOCKS=${VIZ_BLOCKS:-0,4,8,12,16,20,24,27}
VIZ_SELECTED_BLOCKS=${VIZ_SELECTED_BLOCKS:-8,12,16,20}

export DROID_SUCCESS_V21_TAVID_DIR="$EVAL_DATASET_DIR"
export DROID_SUCCESS_V21_TAVID_VAL_DIR="$EVAL_DATASET_DIR"
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2,3,4}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

mkdir -p "$TEXT_ATTN_OUT" "$FEATURE_ATTN_OUT" "$LOG_DIR"

NUM_SCENES=$(python - <<PY
import json
from pathlib import Path
manifest = Path("$EVAL_DATASET_DIR") / "one_per_scene_manifest.json"
print(json.load(open(manifest))["num_written"])
PY
)

echo "date=$(date)"
echo "run_root=$RUN_ROOT"
echo "eval_dataset=$EVAL_DATASET_DIR"
echo "checkpoint=$ATTN_CHECKPOINT"
echo "num_scenes=$NUM_SCENES"
echo "blocks=$VIZ_BLOCKS selected=$VIZ_SELECTED_BLOCKS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

torchrun --standalone --nproc_per_node=1 -m scripts.visualize_tavid_cross_attention \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$ATTN_CHECKPOINT" \
  --output-dir "$TEXT_ATTN_OUT" \
  --split val \
  --num-samples "$NUM_SCENES" \
  --max-batches "$NUM_SCENES" \
  --blocks "$VIZ_BLOCKS" \
  --selected-blocks "$VIZ_SELECTED_BLOCKS" \
  --token-source text \
  --sample-label "text_tgt_iter10000" \
  -- experiment="$EXPERIMENT" \
  dataloader_val.batch_size=1 \
  dataloader_val.num_workers=1 \
  dataloader_val.drop_last=False \
  model.config.net.tavid_attn_query_chunk_size=1024 \
  2>&1 | tee "$LOG_DIR/04b_visualize_text_attention_dcp.log"

torchrun --standalone --nproc_per_node=1 -m scripts.visualize_tavid_cross_attention \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$ATTN_CHECKPOINT" \
  --output-dir "$FEATURE_ATTN_OUT" \
  --split val \
  --num-samples "$NUM_SCENES" \
  --max-batches "$NUM_SCENES" \
  --blocks "$VIZ_BLOCKS" \
  --selected-blocks "$VIZ_SELECTED_BLOCKS" \
  --token-source feature \
  --dummy-text-embeddings \
  --sample-label "instructsam_feature_iter10000" \
  -- experiment="$EXPERIMENT" \
  dataloader_val.batch_size=1 \
  dataloader_val.num_workers=1 \
  dataloader_val.drop_last=False \
  model.config.net.tavid_attn_query_chunk_size=1024 \
  2>&1 | tee "$LOG_DIR/05b_visualize_instructsam_feature_attention_dcp.log"

python - <<PY
import json
from pathlib import Path
run_root = Path("$RUN_ROOT")
summary = {
    "run_root": str(run_root),
    "dataset": "$EVAL_DATASET_DIR",
    "generation": str(run_root / "generation"),
    "text_target_attention": "$TEXT_ATTN_OUT",
    "instructsam_feature_attention": "$FEATURE_ATTN_OUT",
    "num_scenes": int("$NUM_SCENES"),
    "attention_checkpoint": "$ATTN_CHECKPOINT",
}
(run_root / "attention_only_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
