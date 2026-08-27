#!/usr/bin/env bash
# Oracle test: train Cosmos to consume GT future SigLIP2 semantic blueprints.
# The semantic plan is precomputed from future frames as [6,81,1152] and loaded
# as target_feature. No explicit mask is provided to Cosmos.

#SBATCH --job-name=cosmos-siglip2-oracle
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-oracle-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-oracle-%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT" || exit 2
mkdir -p /data/user/jhe724/workspace/VLM4WAM/logs

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1
export PIP_INDEX_URL=${PIP_INDEX_URL:-http://harbor.internal.com:8081/repository/pypi-hkust/simple}
export PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-harbor.internal.com}

export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://10.12.1.245:8080}
export WANDB_API_KEY=${WANDB_API_KEY:-local-37151658708fac20809135dce9e234842db32f97}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm4wam-textfree-multisource}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-siglip2_semantic_oracle_context}

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-1,2,3}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_v21_siglip2_semantic_oracle_context_320_49f_s123_vlm4wam}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

if ! python - <<'PY'
import sys
import transformers

sys.exit(0 if transformers.__version__ == "4.51.3" else 1)
PY
then
  python -m pip install --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" --upgrade \
    transformers==4.51.3 \
    tokenizers==0.21.4 \
    peft==0.17.1 \
    accelerate==1.11.0 || {
      echo "Failed to restore Cosmos-compatible Python packages from ${PIP_INDEX_URL}" >&2
      exit 1
    }
fi

nvidia-smi -L
python -c "import torch; print('cuda count:', torch.cuda.device_count())"
echo "=== code fingerprint (as seen by compute node) ==="
md5sum \
  cosmos_predict2/_src/predict2/datasets/local_datasets/dataset_video.py \
  cosmos_predict2/_src/predict2/models/video2world_model_rectified_flow.py \
  cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py \
  cosmos_predict2/experiments/base/robointer.py

python - <<'PY'
import os
import sys
import json
from pathlib import Path

import torch

d = Path(os.environ["DROID_SUCCESS_V21_TAVID_DIR"])
labels = d / "siglip2_semantic_plan_k6_g9_full"
frame_ranges_path = d / "frame_ranges.json"
if not frame_ranges_path.exists():
    raise RuntimeError(f"Missing frame_ranges manifest: {frame_ranges_path}")
active_stems = sorted(json.loads(frame_ranges_path.read_text()).keys())
if not active_stems:
    raise RuntimeError(f"No frame ranges under {frame_ranges_path}")
if not labels.is_dir():
    raise RuntimeError(f"Missing semantic label dir: {labels}")
missing = [stem for stem in active_stems if not (labels / f"{stem}.pt").exists()]
frac = len(missing) / len(active_stems)
print(f"siglip2_semantic_plan_k6_g9_full: missing={len(missing)} frame_ranges={len(active_stems)} ({frac:.4f})")
if missing:
    print("first missing:", missing[:10])
    sys.exit(1)
payload = torch.load(labels / f"{active_stems[0]}.pt", map_location="cpu", weights_only=False)
plan = payload["semantic_plan"] if isinstance(payload, dict) else payload
print(f"sample={active_stems[0]} semantic_plan shape={tuple(plan.shape)} dtype={plan.dtype}")
assert tuple(plan.shape) == (6, 81, 1152), tuple(plan.shape)
PY

GRAD_ACCUM_ITER=${GRAD_ACCUM_ITER:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_ITER=${MAX_ITER:-1600}
RUN_VALIDATION=${RUN_VALIDATION:-False}
VALIDATION_ITER=${VALIDATION_ITER:-400}
MAX_VAL_ITER=${MAX_VAL_ITER:-64}
RUN_VALIDATION_ON_START=${RUN_VALIDATION_ON_START:-False}
SAVE_ITER=${SAVE_ITER:-400}
SAMPLE_ITER=${SAMPLE_ITER:-400}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-12}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-4}
TAVID_ATTN_QUERY_CHUNK_SIZE=${TAVID_ATTN_QUERY_CHUNK_SIZE:-1024}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_siglip2_semantic_oracle_context_320}
JOB_NAME=${JOB_NAME:-2b_siglip2_semantic_oracle_context_iou50_320_49f_s123_bs4accum4_gbs128_1600}
export WANDB_NAME=${WANDB_NAME:-$JOB_NAME}

echo "=== wandb: mode=${WANDB_MODE} base=${WANDB_BASE_URL} project=${WANDB_PROJECT} name=${WANDB_NAME} ==="
echo "=== TRAIN GT SigLIP2 semantic oracle context; experiment=${EXPERIMENT}; frames=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES}; strides=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES}; per_gpu_batch=${BATCH_SIZE}; grad_accum=${GRAD_ACCUM_ITER}; global_batch=$((BATCH_SIZE * 8 * GRAD_ACCUM_ITER)); max_iter=${MAX_ITER}; job_name=${JOB_NAME}; output_root=${IMAGINAIRE_OUTPUT_ROOT} ==="
torchrun --standalone --nproc_per_node=8 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- experiment="$EXPERIMENT" \
  job.name="$JOB_NAME" \
  job.wandb_mode=disabled \
  "~trainer.callbacks.wandb" \
  "~trainer.callbacks.wandb_10x" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  dataloader_train.num_workers="$TRAIN_NUM_WORKERS" \
  dataloader_val.num_workers="$VAL_NUM_WORKERS" \
  checkpoint.save_iter="$SAVE_ITER" \
  trainer.grad_accum_iter="$GRAD_ACCUM_ITER" \
  trainer.max_iter="$MAX_ITER" \
  trainer.run_validation="$RUN_VALIDATION" \
  trainer.validation_iter="$VALIDATION_ITER" \
  trainer.max_val_iter="$MAX_VAL_ITER" \
  trainer.run_validation_on_start="$RUN_VALIDATION_ON_START" \
  trainer.callbacks.every_n_sample_reg.every_n="$SAMPLE_ITER" \
  trainer.callbacks.every_n_sample_ema.every_n="$SAMPLE_ITER" \
  model.config.net.tavid_attn_query_chunk_size="$TAVID_ATTN_QUERY_CHUNK_SIZE"
status=$?
echo "train_exit=$status"
exit "$status"
