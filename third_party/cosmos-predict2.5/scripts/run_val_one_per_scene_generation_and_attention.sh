#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/cosmos-predict2.5}
VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM}
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

SOURCE_VAL_DIR=${SOURCE_VAL_DIR:-/data/LFT-W02_data/junjie/datasets/droid_success_v21_target_aware_left_right_480x864_val}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/outputs/val_one_per_scene_instructsam_feature_iter10000_$(date +%Y%m%d_%H%M%S)}
EVAL_DATASET_DIR=${EVAL_DATASET_DIR:-$RUN_ROOT/dataset_one_per_scene}
GEN_OUT=${GEN_OUT:-$RUN_ROOT/generation}
TEXT_ATTN_OUT=${TEXT_ATTN_OUT:-$RUN_ROOT/text_target_attention}
FEATURE_ATTN_OUT=${FEATURE_ATTN_OUT:-$RUN_ROOT/instructsam_feature_attention}
LOG_DIR=${LOG_DIR:-$RUN_ROOT/logs}

GEN_CHECKPOINT=${GEN_CHECKPOINT:-$REPO_ROOT/outputs/pulled_checkpoints/instructsam_iter000010000_dcp}
ATTN_CHECKPOINT=${ATTN_CHECKPOINT:-$REPO_ROOT/outputs/pulled_checkpoints/instructsam_iter000010000_pt/model_ema_bf16.pt}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
SEED=${SEED:-20260603}
FPS=${FPS:-8}
VIZ_BLOCKS=${VIZ_BLOCKS:-0,4,8,12,16,20,24,27}
VIZ_SELECTED_BLOCKS=${VIZ_SELECTED_BLOCKS:-8,12,16,20}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

INSTRUCTSAM_PYTHON=${INSTRUCTSAM_PYTHON:-/data/LFT-W02_data/.conda/envs/instructsam/bin/python}
INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/InstructSAM}
INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/LFT-W02_data/junjie/weights/CircleRadon/InstructSAM-2B}
INSTRUCTSAM_EXTRA_PYTHONPATH=${INSTRUCTSAM_EXTRA_PYTHONPATH:-/tmp/instructsam_deps}

mkdir -p "$RUN_ROOT" "$GEN_OUT" "$TEXT_ATTN_OUT" "$FEATURE_ATTN_OUT" "$LOG_DIR"

echo "date=$(date)"
echo "host=$(hostname)"
echo "repo=$REPO_ROOT"
echo "source_val=$SOURCE_VAL_DIR"
echo "run_root=$RUN_ROOT"
echo "eval_dataset=$EVAL_DATASET_DIR"
echo "gen_checkpoint=$GEN_CHECKPOINT"
echo "attn_checkpoint=$ATTN_CHECKPOINT"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

python scripts/build_one_per_scene_eval_subset.py \
  --source-dir "$SOURCE_VAL_DIR" \
  --output-dir "$EVAL_DATASET_DIR" \
  --prefer-camera left_external \
  --overwrite \
  2>&1 | tee "$LOG_DIR/01_build_subset.log"

NUM_SCENES=$(python - <<PY
import json
data=json.load(open("$EVAL_DATASET_DIR/one_per_scene_manifest.json"))
print(data["num_written"])
PY
)
echo "num_scenes=$NUM_SCENES"

PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-cuda:$REPO_ROOT/packages/cosmos-oss:$INSTRUCTSAM_SOURCE_ROOT:$INSTRUCTSAM_EXTRA_PYTHONPATH:${PYTHONPATH:-}" \
"$INSTRUCTSAM_PYTHON" scripts/precompute_instructsam_target_features.py \
  --dataset-dir "$EVAL_DATASET_DIR" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --query-template "Please segment '{target}' in the image." \
  --feature-mode mask_query \
  --combine-mode best \
  --skip-existing \
  --fallback-zero-on-missing-feature \
  --max-errors 20 \
  --log-every 5 \
  2>&1 | tee "$LOG_DIR/02_precompute_instructsam_features.log"

export DROID_SUCCESS_V21_TAVID_DIR="$EVAL_DATASET_DIR"
export DROID_SUCCESS_V21_TAVID_VAL_DIR="$EVAL_DATASET_DIR"
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2,3,4}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$GEN_CHECKPOINT" \
  --output-dir "$GEN_OUT" \
  --num-samples "$NUM_SCENES" \
  --num-steps "$NUM_STEPS" \
  --guidance "$GUIDANCE" \
  --seed "$SEED" \
  --fps "$FPS" \
  --max-batches "$NUM_SCENES" \
  --standalone-only \
  -- experiment="$EXPERIMENT" \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2 \
  dataloader_train.drop_last=False \
  dataloader_train.dataset.target_mask_dropout_prob=0.0 \
  dataloader_train.dataset.target_mask_default_to_zero=False \
  dataloader_train.dataset.target_feature_default_to_zero=False \
  trainer.grad_accum_iter=1 \
  trainer.run_validation=False \
  2>&1 | tee "$LOG_DIR/03_generate_videos.log"

torchrun --standalone --nproc_per_node=1 -m scripts.visualize_tavid_cross_attention \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$ATTN_CHECKPOINT" \
  --output-dir "$TEXT_ATTN_OUT" \
  --split val \
  --num-samples "$NUM_SCENES" \
  --max-batches "$NUM_SCENES" \
  --blocks "$VIZ_BLOCKS" \
  --selected-blocks "$VIZ_SELECTED_BLOCKS" \
  --model-only-load \
  --skip-init-environment \
  --token-source text \
  --sample-label "text_tgt_iter10000" \
  -- experiment="$EXPERIMENT" \
  dataloader_val.batch_size=1 \
  dataloader_val.num_workers=0 \
  dataloader_val.drop_last=False \
  model.config.net.tavid_attn_query_chunk_size=1024 \
  2>&1 | tee "$LOG_DIR/04_visualize_text_attention.log"

torchrun --standalone --nproc_per_node=1 -m scripts.visualize_tavid_cross_attention \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$ATTN_CHECKPOINT" \
  --output-dir "$FEATURE_ATTN_OUT" \
  --split val \
  --num-samples "$NUM_SCENES" \
  --max-batches "$NUM_SCENES" \
  --blocks "$VIZ_BLOCKS" \
  --selected-blocks "$VIZ_SELECTED_BLOCKS" \
  --model-only-load \
  --skip-init-environment \
  --token-source feature \
  --dummy-text-embeddings \
  --sample-label "instructsam_feature_iter10000" \
  -- experiment="$EXPERIMENT" \
  dataloader_val.batch_size=1 \
  dataloader_val.num_workers=0 \
  dataloader_val.drop_last=False \
  model.config.net.tavid_attn_query_chunk_size=1024 \
  2>&1 | tee "$LOG_DIR/05_visualize_instructsam_feature_attention.log"

python - <<PY
import json
from pathlib import Path
run_root = Path("$RUN_ROOT")
summary = {
    "run_root": str(run_root),
    "dataset": "$EVAL_DATASET_DIR",
    "generation": "$GEN_OUT",
    "text_target_attention": "$TEXT_ATTN_OUT",
    "instructsam_feature_attention": "$FEATURE_ATTN_OUT",
    "logs": "$LOG_DIR",
    "num_scenes": int("$NUM_SCENES"),
    "gen_checkpoint": "$GEN_CHECKPOINT",
    "attn_checkpoint": "$ATTN_CHECKPOINT",
}
(run_root / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
