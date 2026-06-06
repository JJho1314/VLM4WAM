#!/usr/bin/env bash
# Precompute InstructSAM target features for strict holdout v3 DROID success data.

#SBATCH --job-name=precomp-isam-v3
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-precompute-instructsam-v3-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-precompute-instructsam-v3-%j.err

set -euo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1

export WANDB_MODE=online
export WANDB_BASE_URL="http://10.12.1.245:8080"
export WANDB_API_KEY="local-37151658708fac20809135dce9e234842db32f97"

export INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/InstructSAM}
export INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/InstructSAM/work_dirs/InstructSAM-2B}
export INSTRUCTSAM_TRANSFORMERS_ROOT=${INSTRUCTSAM_TRANSFORMERS_ROOT:-/data/user/jhe724/workspace/transformers-instructsam-9269c1b}
export PYTHONPATH="$INSTRUCTSAM_TRANSFORMERS_ROOT/src:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$DROID_SUCCESS_V21_TAVID_DIR/target_features" "$DROID_SUCCESS_V21_TAVID_VAL_DIR/target_features"

python - <<'PY'
import transformers
print("precompute_transformers:", transformers.__version__, transformers.__file__)
import transformers.models.qwen3_vl.video_processing_qwen3_vl  # noqa: F401
from instructsam.models import load_pretrained_model  # noqa: F401
PY

nvidia-smi -L
python - <<'PY'
import os
from pathlib import Path


def count_active(dataset_dir):
    dataset_dir = Path(dataset_dir)
    videos = sorted((dataset_dir / "videos").glob("*.mp4"))
    exclude_path = dataset_dir / "exclude_no_tgt_stems.txt"
    excluded = set(exclude_path.read_text().split()) if exclude_path.exists() else set()
    active = [path for path in videos if path.stem not in excluded]
    features = sorted((dataset_dir / "target_features").glob("*.pt"))
    print("dataset:", dataset_dir)
    print("videos_raw:", len(videos))
    print("excluded_no_tgt:", len(excluded))
    print("videos_active:", len(active))
    print("features_existing:", len(features))


print("instructsam_source:", os.environ["INSTRUCTSAM_SOURCE_ROOT"], Path(os.environ["INSTRUCTSAM_SOURCE_ROOT"]).exists())
print("instructsam_model:", os.environ["INSTRUCTSAM_MODEL_PATH"], Path(os.environ["INSTRUCTSAM_MODEL_PATH"]).exists())
count_active(os.environ["DROID_SUCCESS_V21_TAVID_DIR"])
count_active(os.environ["DROID_SUCCESS_V21_TAVID_VAL_DIR"])
PY

torchrun --standalone --nproc_per_node=8 scripts/precompute_instructsam_target_features.py \
  --dataset-dir "$DROID_SUCCESS_V21_TAVID_DIR" \
  --dataset-dir "$DROID_SUCCESS_V21_TAVID_VAL_DIR" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --output-dir-name target_features \
  --feature-mode mask_query \
  --expected-feature-dim 256 \
  --combine-mode best \
  --fallback-zero-on-missing-feature \
  --fallback-zero-tokens 64 \
  --skip-existing \
  --log-every 25

python - <<'PY'
import os
import sys
from pathlib import Path


def validate(dataset_dir):
    dataset_dir = Path(dataset_dir)
    videos = sorted((dataset_dir / "videos").glob("*.mp4"))
    exclude_path = dataset_dir / "exclude_no_tgt_stems.txt"
    excluded = set(exclude_path.read_text().split()) if exclude_path.exists() else set()
    active = [path for path in videos if path.stem not in excluded]
    missing = [path.stem for path in active if not (dataset_dir / "target_features" / f"{path.stem}.pt").exists()]
    print(f"validate dataset={dataset_dir} active={len(active)} missing_features={len(missing)}")
    if missing:
        print("first_missing:", missing[:20])
        return False
    return True


ok = validate(os.environ["DROID_SUCCESS_V21_TAVID_DIR"])
ok = validate(os.environ["DROID_SUCCESS_V21_TAVID_VAL_DIR"]) and ok
sys.exit(0 if ok else 1)
PY
