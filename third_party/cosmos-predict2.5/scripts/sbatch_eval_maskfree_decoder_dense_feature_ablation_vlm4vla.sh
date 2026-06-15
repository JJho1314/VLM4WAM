#!/usr/bin/env bash
# Evaluate whether mask-free InstructSAM decoder-dense features affect Cosmos generation.

#SBATCH --job-name=cosmos-mf-eval
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4VLA/slurm-maskfree-eval-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4VLA/slurm-maskfree-eval-%j.err

set -uo pipefail

VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jhe724/workspace/VLM4VLA}
REPO_ROOT=${REPO_ROOT:-$VLM4VLA_ROOT/third_party/cosmos-predict2.5}
cd "$REPO_ROOT" || exit 2

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=${VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV="$VENV"
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-cuda:$REPO_ROOT/packages/cosmos-oss:${PYTHONPATH:-}"

export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/user/jhe724/workspace/weights}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export COSMOS_SKIP_CUDA_VERSION_CHECK=${COSMOS_SKIP_CUDA_VERSION_CHECK:-1}

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-$DROID_SUCCESS_V21_TAVID_DIR}
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2,3,4}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

CHECKPOINT=${CHECKPOINT:-$VLM4VLA_ROOT/outputs/droid_success_v21_maskfree_decoder_dense_stage2_lora_scene_cap200_tasktarget/cosmos_predict_v2p5/video2world/2b_droid_success_v21_maskfree_decoder_dense_stage2_lora_scene_cap200_tasktarget_49f_s234_bs2accum8_gbs128_2000step_val400_from_base/checkpoints/iter_000002000}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_maskfree_decoder_dense_target}
FEATURE_DIR=${FEATURE_DIR:-$DROID_SUCCESS_V21_TAVID_DIR/target_features_instructsam_decoder_dense_stage2_lora}
RUN_ROOT=${RUN_ROOT:-$VLM4VLA_ROOT/outputs/eval_maskfree_decoder_dense_iter2000_feature_ablation_$(date +%Y%m%d_%H%M%S)}
SEED=${SEED:-20260612}
NUM_STEPS=${NUM_STEPS:-20}
GUIDANCE=${GUIDANCE:-3.0}
FPS=${FPS:-8}
NUM_SAMPLES=${NUM_SAMPLES:-1}
MAX_BATCHES=${MAX_BATCHES:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

mkdir -p "$RUN_ROOT/logs"

WRONG_FEATURE=$(find "$FEATURE_DIR" -maxdepth 1 -type f -name '*.pt' | sort | sed -n '2p')
if [ -z "$WRONG_FEATURE" ]; then
  echo "No wrong feature candidate found in $FEATURE_DIR" >&2
  exit 3
fi

{
  date
  hostname
  echo "repo=$REPO_ROOT"
  echo "checkpoint=$CHECKPOINT"
  echo "experiment=$EXPERIMENT"
  echo "dataset=$DROID_SUCCESS_V21_TAVID_DIR"
  echo "feature_dir=$FEATURE_DIR"
  echo "wrong_feature=$WRONG_FEATURE"
  echo "run_root=$RUN_ROOT"
  echo "seed=$SEED num_steps=$NUM_STEPS guidance=$GUIDANCE num_samples=$NUM_SAMPLES max_batches=$MAX_BATCHES"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
} | tee "$RUN_ROOT/logs/00_run_info.log"

run_variant() {
  local name=$1
  local mode=$2
  local feature_path=${3:-}
  local out="$RUN_ROOT/$name"
  local log="$RUN_ROOT/logs/${name}.log"
  mkdir -p "$out"
  local extra=()
  if [ "$mode" = "path" ]; then
    extra=(--target-feature-path "$feature_path")
  fi
  set +e
  torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
    --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$out" \
    --num-samples "$NUM_SAMPLES" \
    --num-steps "$NUM_STEPS" \
    --guidance "$GUIDANCE" \
    --seed "$SEED" \
    --fps "$FPS" \
    --max-batches "$MAX_BATCHES" \
    --standalone-only \
    --reuse-encoded-latent \
    --offload-denoiser-during-vae \
    --offload-denoiser-before-decode \
    --allow-empty-target-mask \
    --remove-target-mask \
    --target-feature-mode "$mode" \
    "${extra[@]}" \
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
run_variant wrong_feature path "$WRONG_FEATURE" || overall=1

if [ "$overall" -eq 0 ]; then
  python scripts/analyze_target_feature_ablation.py \
    --run-root "$RUN_ROOT" \
    --variants keep zero drop wrong_feature \
    --output-prefix maskfree_decoder_dense_iter2000 \
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
    "dataset": "$DROID_SUCCESS_V21_TAVID_DIR",
    "feature_dir": "$FEATURE_DIR",
    "wrong_feature": "$WRONG_FEATURE",
    "seed": int("$SEED"),
    "num_steps": int("$NUM_STEPS"),
    "guidance": float("$GUIDANCE"),
    "variants": ["keep", "zero", "drop", "wrong_feature"],
    "overall_exit": int("$overall"),
}
(root / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

exit "$overall"
