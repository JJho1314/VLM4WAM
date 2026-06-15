#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT" || exit 2

COSMOS_VENV=${COSMOS_VENV:-/data/LFT-W02_data/junjie/cosmos-predict2.5/.venv}
export VIRTUAL_ENV="$COSMOS_VENV"
export PATH="$COSMOS_VENV/bin:$PATH"
unset PYTHONHOME

NV_LIB=$COSMOS_VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-cuda:$REPO_ROOT/packages/cosmos-oss:${PYTHONPATH:-}"
export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/LFT-W02_data/junjie/weights}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export COSMOS_SKIP_CUDA_VERSION_CHECK=${COSMOS_SKIP_CUDA_VERSION_CHECK:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/outputs/tavid_generation_runs/robointer_74616_yellow_carrot_prompt_targetaware_dataset}
FEATURE_DIR=${FEATURE_DIR:-$DATASET_DIR/target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt}
CHECKPOINT=${CHECKPOINT:-$REPO_ROOT/outputs/pulled_checkpoints/latent_grounding_iter_000000400}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_latent_grounding_decoder_dense_target}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/outputs/eval_latent_grounding_iter400_yellow_carrot_green_leaf_local_$(date +%Y%m%d_%H%M%S)}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-latent_grounding_iter400_yellow_carrot_green_leaf}
SEED=${SEED:-20260613}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
FPS=${FPS:-8}

export DROID_SUCCESS_V21_TAVID_DIR="$DATASET_DIR"
export DROID_SUCCESS_V21_TAVID_VAL_DIR="$DATASET_DIR"
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-1}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

mkdir -p "$RUN_ROOT/logs"
{
  date
  hostname
  echo "repo=$REPO_ROOT"
  echo "checkpoint=$CHECKPOINT"
  echo "experiment=$EXPERIMENT"
  echo "dataset=$DATASET_DIR"
  echo "feature_dir=$FEATURE_DIR"
  echo "run_root=$RUN_ROOT"
  echo "seed=$SEED num_steps=$NUM_STEPS guidance=$GUIDANCE"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
} | tee "$RUN_ROOT/logs/00_run_info.log"

run_variant() {
  local name=$1
  local mode=$2
  local out="$RUN_ROOT/$name"
  local log="$RUN_ROOT/logs/${name}.log"
  mkdir -p "$out"
  set +e
  torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
    --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$out" \
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
    --allow-empty-target-mask \
    --remove-target-mask \
    --target-feature-mode "$mode" \
    -- experiment="$EXPERIMENT" \
    dataloader_train.batch_size=1 \
    dataloader_train.num_workers=2 \
    dataloader_train.drop_last=False \
    dataloader_train.dataset.target_mask_dir=none \
    dataloader_train.dataset.target_mask_default_to_zero=True \
    dataloader_train.dataset.target_feature_dir="$FEATURE_DIR" \
    dataloader_train.dataset.target_feature_default_to_zero=False \
    trainer.grad_accum_iter=1 \
    trainer.run_validation=False \
    2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$RUN_ROOT/logs/${name}.exit"
  return "$status"
}

overall=0
run_variant keep keep || overall=1
run_variant zero zero || overall=1
run_variant drop drop || overall=1

if [ "$overall" -eq 0 ]; then
  python scripts/analyze_target_feature_ablation.py \
    --run-root "$RUN_ROOT" \
    --variants keep zero drop \
    --mask-npz "$DATASET_DIR/target_masks/74616_exterior_image_1_left.npz" \
    --output-prefix "$OUTPUT_PREFIX" \
    2>&1 | tee "$RUN_ROOT/logs/analyze.log" || overall=1
fi

python - <<PY
import json
from pathlib import Path
root = Path("$RUN_ROOT")
summary = {
    "run_root": str(root),
    "checkpoint": "$CHECKPOINT",
    "experiment": "$EXPERIMENT",
    "dataset": "$DATASET_DIR",
    "feature_dir": "$FEATURE_DIR",
    "seed": int("$SEED"),
    "num_steps": int("$NUM_STEPS"),
    "guidance": float("$GUIDANCE"),
    "variants": ["keep", "zero", "drop"],
    "cosmos_target_mask_removed": True,
    "overall_exit": int("$overall"),
    "contact_sheet": str(root / ("$OUTPUT_PREFIX" + "_contact_sheet.jpg")),
    "diff_sheet": str(root / ("$OUTPUT_PREFIX" + "_diff_vs_keep.jpg")),
    "metrics": str(root / ("$OUTPUT_PREFIX" + "_metrics.json")),
}
(root / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

exit "$overall"
