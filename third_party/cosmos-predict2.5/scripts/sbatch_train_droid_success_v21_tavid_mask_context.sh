#!/usr/bin/env bash
# Train Cosmos 2B TAViD-style target-aware WITHOUT mask-in-latent: text + [TGT] +
# TAVID attention loss + mask as SPATIAL context tokens (TargetMaskContextAdapter).
# Defaults to the cap200_tasktarget set, same budget as the textfree run.

#SBATCH --job-name=cosmos-tavid-mctx
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-tavid-mctx-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-tavid-mctx-%j.err

set -uo pipefail
# NOTE: Slurm spools the batch script; use the deploy path (overridable).
REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"
mkdir -p /data/user/jhe724/workspace/VLM4WAM/logs

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"   # this repo's cosmos_predict2 wins

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1
export PIP_INDEX_URL=${PIP_INDEX_URL:-http://harbor.internal.com:8081/repository/pypi-hkust/simple}
export PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-harbor.internal.com}

# Internal (intranet) wandb.
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://10.12.1.245:8080}
export WANDB_API_KEY=${WANDB_API_KEY:-local-37151658708fac20809135dce9e234842db32f97}
export WANDB_PROJECT=${WANDB_PROJECT:-vlm4wam-textfree-multisource}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-tavid_mask_context}

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_scene_cap200_tasktarget}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2,3,4}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_tavid_mask_context_cap200_tasktarget}
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
python - <<'PY'
import os
import sys
from pathlib import Path

# This experiment trains on GT masks: require mask coverage on the train dir.
TOLERATE = 0.01
d = Path(os.environ["DROID_SUCCESS_V21_TAVID_DIR"])
videos = sorted((d / "videos").glob("*.mp4"))
exclude = d / "exclude_no_tgt_stems.txt"
excluded = set(exclude.read_text().split()) if exclude.exists() else set()
active = [p for p in videos if p.stem not in excluded]
mask_dir = d / "masks"
missing = [p.stem for p in active if not any((mask_dir / f"{p.stem}{ext}").exists() for ext in (".npz", ".npy", ".png"))]
frac = (len(missing) / len(active)) if active else 0.0
print(f"train dir={d} active={len(active)} missing_masks={len(missing)} frac={frac:.4f}")
if missing:
    print("first_missing:", missing[:10])
if frac > TOLERATE:
    sys.exit(1)
PY

GRAD_ACCUM_ITER=${GRAD_ACCUM_ITER:-8}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_ITER=${MAX_ITER:-1370}
RUN_VALIDATION=${RUN_VALIDATION:-False}
VALIDATION_ITER=${VALIDATION_ITER:-1000}
MAX_VAL_ITER=${MAX_VAL_ITER:-64}
RUN_VALIDATION_ON_START=${RUN_VALIDATION_ON_START:-False}
SAVE_ITER=${SAVE_ITER:-274}
SAMPLE_ITER=${SAMPLE_ITER:-137}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-12}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-4}
TAVID_ATTN_QUERY_CHUNK_SIZE=${TAVID_ATTN_QUERY_CHUNK_SIZE:-1024}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_tavid_mask_context}
JOB_NAME=${JOB_NAME:-2b_droid_success_v21_tavid_mask_context_cap200_49f_bs2accum8_1370}
export WANDB_NAME=${WANDB_NAME:-$JOB_NAME}

echo "=== wandb: mode=${WANDB_MODE} base=${WANDB_BASE_URL} project=${WANDB_PROJECT} name=${WANDB_NAME} ==="
echo "=== TRAIN TAViD mask-context (no mask-in-latent); experiment=${EXPERIMENT}; per_gpu_batch=${BATCH_SIZE}; grad_accum=${GRAD_ACCUM_ITER}; global_batch=$((BATCH_SIZE * 8 * GRAD_ACCUM_ITER)); max_iter=${MAX_ITER}; job_name=${JOB_NAME}; output_root=${IMAGINAIRE_OUTPUT_ROOT} ==="
torchrun --standalone --nproc_per_node=8 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- experiment="$EXPERIMENT" \
  job.name="$JOB_NAME" \
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
