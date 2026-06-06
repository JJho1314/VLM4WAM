#!/usr/bin/env bash
# Train Cosmos 2B with implicit InstructSAM target features in an isolated VLM4VLA workspace.

#SBATCH --job-name=cosmos-isam-vlm4vla
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4VLA/slurm-instructsam-v3-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4VLA/slurm-instructsam-v3-%j.err

set -uo pipefail

# Slurm executes a copied script from its spool directory, so BASH_SOURCE cannot
# reliably recover the original checkout. Keep the isolated repo explicit.
REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4VLA/third_party/cosmos-predict2.5}
VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jhe724/workspace/VLM4VLA}
if [ ! -f "$REPO_ROOT/scripts/train.py" ]; then
  echo "Invalid REPO_ROOT=${REPO_ROOT}; scripts/train.py not found." >&2
  exit 2
fi
cd "$REPO_ROOT"

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

# Reuse the known-good environment, but do not auto-modify it from this isolated experiment.
VENV=${VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/user/jhe724/workspace/weights}
export HF_HUB_OFFLINE=1
export PIP_INDEX_URL=${PIP_INDEX_URL:-http://harbor.internal.com:8081/repository/pypi-hkust/simple}
export PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST:-harbor.internal.com}

export WANDB_MODE=online
export WANDB_BASE_URL="http://10.12.1.245:8080"
export WANDB_API_KEY="local-37151658708fac20809135dce9e234842db32f97"

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2,3,4}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$VLM4VLA_ROOT/outputs/droid_success_v21_instructsam_feature_context_strict_holdout_v3}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$VLM4VLA_ROOT" "$IMAGINAIRE_OUTPUT_ROOT"

if ! python - <<'PY'
import sys
import transformers

sys.exit(0 if transformers.__version__ == "4.51.3" else 1)
PY
then
  echo "Shared VENV has incompatible transformers version; refusing to auto-modify ${VENV}." >&2
  echo "Fix the environment explicitly or pass VENV pointing at an isolated environment." >&2
  exit 1
fi

nvidia-smi -L
python -c "import torch; print('cuda count:', torch.cuda.device_count())"
python - <<'PY'
import os
import sys
from pathlib import Path


def count_active(dataset_dir):
    dataset_dir = Path(dataset_dir)
    videos = sorted((dataset_dir / "videos").glob("*.mp4"))
    exclude_path = dataset_dir / "exclude_no_tgt_stems.txt"
    excluded = set(exclude_path.read_text().split()) if exclude_path.exists() else set()
    active = [path for path in videos if path.stem not in excluded]
    masks = sorted((dataset_dir / "masks").glob("*"))
    features = sorted((dataset_dir / "target_features").glob("*.pt"))
    missing = [path.stem for path in active if not (dataset_dir / "target_features" / f"{path.stem}.pt").exists()]
    return len(videos), len(excluded), len(active), len(masks), len(features), missing


for key in ("DROID_SUCCESS_V21_TAVID_DIR", "DROID_SUCCESS_V21_TAVID_VAL_DIR"):
    raw, excluded, active, masks, features, missing = count_active(os.environ[key])
    print(key, os.environ[key])
    print("videos_raw:", raw)
    print("excluded_no_tgt:", excluded)
    print("videos_active:", active)
    print("masks:", masks)
    print("target_features:", features)
    print("missing_target_features:", len(missing))
    if missing:
        print("first_missing:", missing[:20])
        sys.exit(1)

print("num_frames:", os.environ["DROID_SUCCESS_V21_TAVID_NUM_FRAMES"])
print("frame_strides:", os.environ["DROID_SUCCESS_V21_TAVID_FRAME_STRIDES"])
print("frame_start_policy:", os.environ["DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY"])
PY

GRAD_ACCUM_ITER=${GRAD_ACCUM_ITER:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_ITER=${MAX_ITER:-14000}
RUN_VALIDATION=${RUN_VALIDATION:-True}
VALIDATION_ITER=${VALIDATION_ITER:-1000}
MAX_VAL_ITER=${MAX_VAL_ITER:-64}
RUN_VALIDATION_ON_START=${RUN_VALIDATION_ON_START:-False}
SAVE_ITER=${SAVE_ITER:-1000}
SAMPLE_ITER=${SAMPLE_ITER:-1000}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-12}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-4}
TAVID_ATTN_QUERY_CHUNK_SIZE=${TAVID_ATTN_QUERY_CHUNK_SIZE:-1024}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context}
JOB_NAME=${JOB_NAME:-2b_droid_success_v21_instructsam_feature_context_strict_v3_49f_s234_bs2accum4_14k_val1000_from_base_balancedmass}

echo "=== TRAIN DROID success v21 InstructSAM feature context strict_holdout_v3; repo=${REPO_ROOT}; experiment=${EXPERIMENT}; per_gpu_batch=${BATCH_SIZE}; grad_accum=${GRAD_ACCUM_ITER}; global_batch=$((BATCH_SIZE * 8 * GRAD_ACCUM_ITER)); max_iter=${MAX_ITER}; validation=${RUN_VALIDATION}; validation_iter=${VALIDATION_ITER}; max_val_iter=${MAX_VAL_ITER}; save_iter=${SAVE_ITER}; sample_iter=${SAMPLE_ITER}; train_workers=${TRAIN_NUM_WORKERS}; val_workers=${VAL_NUM_WORKERS}; attn_query_chunk=${TAVID_ATTN_QUERY_CHUNK_SIZE}; job_name=${JOB_NAME}; output_root=${IMAGINAIRE_OUTPUT_ROOT}; wandb_base=${WANDB_BASE_URL} ==="
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
