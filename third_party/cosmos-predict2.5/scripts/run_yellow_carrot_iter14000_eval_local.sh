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

DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/outputs/tavid_generation_runs/robointer_74616_yellow_carrot_prompt_targetaware_dataset}
GEN_CHECKPOINT=${GEN_CHECKPOINT:-$REPO_ROOT/outputs/pulled_checkpoints/instructsam_iter000014000_dcp}
ATTN_CHECKPOINT=${ATTN_CHECKPOINT:-$GEN_CHECKPOINT}
OUT=${OUT:-$REPO_ROOT/outputs/tavid_generation_runs/robointer_74616_iter14000_instructsam_feature_prompt_targetaware_yellow_carrot_to_banana_pot_480p_49f_35step}
ATTN_OUT=${ATTN_OUT:-$REPO_ROOT/outputs/tavid_attention_visualizations/robointer_74616_iter14000_feature_attention_prompt_targetaware_blocks_0_4_8_12_16_20_24_27}
LOG_DIR=${LOG_DIR:-$OUT/logs}

EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
SEED=${SEED:-20260526}
FPS=${FPS:-8}
VIZ_BLOCKS=${VIZ_BLOCKS:-0,4,8,12,16,20,24,27}
VIZ_SELECTED_BLOCKS=${VIZ_SELECTED_BLOCKS:-8,12,16,20}

export DROID_SUCCESS_V21_TAVID_DIR="$DATASET_DIR"
export DROID_SUCCESS_V21_TAVID_VAL_DIR="$DATASET_DIR"
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2,3,4}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

mkdir -p "$OUT" "$ATTN_OUT" "$LOG_DIR"

{
  echo "date=$(date)"
  echo "host=$(hostname)"
  echo "repo=$REPO_ROOT"
  echo "dataset=$DATASET_DIR"
  echo "gen_checkpoint=$GEN_CHECKPOINT"
  echo "attn_checkpoint=$ATTN_CHECKPOINT"
  echo "out=$OUT"
  echo "attn_out=$ATTN_OUT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
} | tee "$LOG_DIR/00_run_info.log"

torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$GEN_CHECKPOINT" \
  --output-dir "$OUT" \
  --num-samples 1 \
  --num-steps "$NUM_STEPS" \
  --guidance "$GUIDANCE" \
  --seed "$SEED" \
  --fps "$FPS" \
  --max-batches 1 \
  --standalone-only \
  --reuse-encoded-latent \
  --offload-denoiser-during-vae \
  --offload-denoiser-before-decode \
  -- experiment="$EXPERIMENT" \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2 \
  dataloader_train.drop_last=False \
  dataloader_train.dataset.target_mask_dropout_prob=0.0 \
  dataloader_train.dataset.target_mask_default_to_zero=False \
  dataloader_train.dataset.target_feature_default_to_zero=False \
  trainer.grad_accum_iter=1 \
  trainer.run_validation=False \
  2>&1 | tee "$LOG_DIR/01_generate_iter14000.log"

torchrun --standalone --nproc_per_node=1 -m scripts.visualize_tavid_cross_attention \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$ATTN_CHECKPOINT" \
  --output-dir "$ATTN_OUT" \
  --split val \
  --num-samples 1 \
  --max-batches 1 \
  --blocks "$VIZ_BLOCKS" \
  --selected-blocks "$VIZ_SELECTED_BLOCKS" \
  --token-source feature \
  --dummy-text-embeddings \
  --offload-denoiser-during-vae \
  --sample-label "instructsam_feature_iter14000" \
  -- experiment="$EXPERIMENT" \
  dataloader_val.batch_size=1 \
  dataloader_val.num_workers=2 \
  dataloader_val.drop_last=False \
  model.config.net.tavid_attn_query_chunk_size=1024 \
  2>&1 | tee "$LOG_DIR/02_visualize_feature_attention_iter14000.log"

INSTRUCTSAM_PYTHON=${INSTRUCTSAM_PYTHON:-/data/LFT-W02_data/.conda/envs/instructsam/bin/python}
INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/InstructSAM}
INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/LFT-W02_data/junjie/weights/CircleRadon/InstructSAM-2B}
INSTRUCTSAM_EXTRA_PYTHONPATH=${INSTRUCTSAM_EXTRA_PYTHONPATH:-/tmp/instructsam_deps}

PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-cuda:$REPO_ROOT/packages/cosmos-oss:$INSTRUCTSAM_SOURCE_ROOT:$INSTRUCTSAM_EXTRA_PYTHONPATH:${PYTHONPATH:-}" \
"$INSTRUCTSAM_PYTHON" scripts/visualize_generated_first_frame_instructsam_masks.py \
  --run-root "$OUT" \
  --attention-dir "$ATTN_OUT" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --torch-dtype bfloat16 \
  --combine-mode best \
  --mask-threshold 0.0 \
  --sample-tile-width 560 \
  --contact-tile-width 560 \
  --font-size 30 \
  --label-height 86 \
  --row-gap 20 \
  --sample-gap 42 \
  2>&1 | tee "$LOG_DIR/03_generated_first_frame_instructsam_check_wrapped.log"

python - <<PY
import json
from pathlib import Path
summary = {
    "out": "$OUT",
    "attention_out": "$ATTN_OUT",
    "visualization": "$OUT/generated_first_frame_instructsam_check/sample_000_generated_first_mask_vs_video.jpg",
    "generation_summary": "$OUT/tavid_generation_summary.json",
    "attention_summary": "$ATTN_OUT/cross_attention_visualization_summary.json",
}
Path("$OUT/run_iter14000_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
