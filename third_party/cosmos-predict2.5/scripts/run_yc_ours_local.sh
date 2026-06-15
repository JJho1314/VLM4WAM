#!/usr/bin/env bash
# Run OUR text-free multisource model on the yellow carrot scene, LOCALLY on the
# A6000, feeding precomputed multi-source target features (carrot/banana/zero)
# via the target_feature_path inference field. Mirrors the local feature_context
# yellow-carrot script's env (local cosmos venv with cosmos_cuda).

set -uo pipefail
REPO_ROOT=${REPO_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
VENV=${VENV:-/data/LFT-W02_data/junjie/cosmos-predict2.5/.venv}
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONHOME
export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/LFT-W02_data/junjie/weights}
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-cuda:$REPO_ROOT/packages/cosmos-oss:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export COSMOS_SKIP_CUDA_VERSION_CHECK=${COSMOS_SKIP_CUDA_VERSION_CHECK:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Existing local dir with videos so the experiment config instantiates (not used).
YC_DS=$REPO_ROOT/outputs/tavid_generation_runs/robointer_74616_yellow_carrot_prompt_targetaware_dataset
export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-$YC_DS}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-$YC_DS}

CKPT=${CKPT:?set CKPT=/path/model_ema_bf16.pt}
SAMPLES=${SAMPLES:?set SAMPLES=/path/samples.jsonl}
OUTDIR=${OUTDIR:?set OUTDIR=/path/videos}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-7.0}

nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
torchrun --standalone --nproc_per_node=1 examples/inference.py \
  -i "$SAMPLES" \
  --output-dir "$OUTDIR" \
  --experiment predict2_video2world_training_2b_droid_success_v21_instructsam_textfree_multisource \
  --checkpoint-path "$CKPT" \
  --config-file cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --disable-guardrails \
  --num-steps "$NUM_STEPS" \
  --guidance "$GUIDANCE"
echo "run_exit=$?"
