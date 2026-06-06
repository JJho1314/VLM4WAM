#!/usr/bin/env bash
# Precompute oracle GT-mask target features for strict holdout v3 DROID success data.

#SBATCH --job-name=gtmask-feat-v3
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4VLA/slurm-gtmask-features-v3-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4VLA/slurm-gtmask-features-v3-%j.err

set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4VLA/third_party/cosmos-predict2.5}
if [ ! -f "$REPO_ROOT/scripts/precompute_gt_mask_target_features.py" ]; then
  echo "Invalid REPO_ROOT=${REPO_ROOT}; precompute script not found." >&2
  exit 2
fi
cd "$REPO_ROOT"

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=${VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/VLM4VLA/third_party/InstructSAM}
export INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/InstructSAM/work_dirs/InstructSAM-2B}
export INSTRUCTSAM_TRANSFORMERS_ROOT=${INSTRUCTSAM_TRANSFORMERS_ROOT:-/data/user/jhe724/workspace/transformers-instructsam-9269c1b}
export PYTHONPATH="$INSTRUCTSAM_TRANSFORMERS_ROOT/src:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}
export TARGET_FEATURE_DIR_NAME=${TARGET_FEATURE_DIR_NAME:-target_features_gt_mask}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$DROID_SUCCESS_V21_TAVID_DIR/$TARGET_FEATURE_DIR_NAME" "$DROID_SUCCESS_V21_TAVID_VAL_DIR/$TARGET_FEATURE_DIR_NAME"

nvidia-smi -L
python - <<'PY'
import os
from pathlib import Path

print("instructsam_source:", os.environ["INSTRUCTSAM_SOURCE_ROOT"], Path(os.environ["INSTRUCTSAM_SOURCE_ROOT"]).exists())
print("instructsam_model:", os.environ["INSTRUCTSAM_MODEL_PATH"], Path(os.environ["INSTRUCTSAM_MODEL_PATH"]).exists())
for key in ("DROID_SUCCESS_V21_TAVID_DIR", "DROID_SUCCESS_V21_TAVID_VAL_DIR"):
    dataset_dir = Path(os.environ[key])
    feature_dir = dataset_dir / os.environ["TARGET_FEATURE_DIR_NAME"]
    videos = sorted((dataset_dir / "videos").glob("*.mp4"))
    masks = sorted((dataset_dir / "masks").glob("*.npz"))
    features = sorted(feature_dir.glob("*.pt"))
    print(key, dataset_dir)
    print("videos:", len(videos), "masks:", len(masks), "features_existing:", len(features), "feature_dir:", feature_dir)
PY

torchrun --standalone --nproc_per_node=8 scripts/precompute_gt_mask_target_features.py \
  --dataset-dir "$DROID_SUCCESS_V21_TAVID_DIR" \
  --dataset-dir "$DROID_SUCCESS_V21_TAVID_VAL_DIR" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --output-dir-name "$TARGET_FEATURE_DIR_NAME" \
  --expected-feature-dim 256 \
  --mask-frame-policy first \
  --skip-existing \
  --log-every 25
