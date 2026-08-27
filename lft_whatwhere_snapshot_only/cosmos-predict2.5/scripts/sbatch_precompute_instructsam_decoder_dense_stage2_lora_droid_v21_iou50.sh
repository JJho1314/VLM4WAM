#!/usr/bin/env bash
# Precompute text-conditioned InstructSAM decoder-dense features for the current
# droid_v21_iou50_taskdiverse_half match-ground dataset.

#SBATCH --job-name=isam-dd-iou50
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-isam-dd-iou50-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-isam-dd-iou50-%j.err

set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
REPO_ROOT=${REPO_ROOT:-$VLM4WAM_ROOT/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"
mkdir -p "$VLM4WAM_ROOT/logs"

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

ISAM_ENV=${ISAM_ENV:-/data/user/jhe724/.conda/envs/instructsam}
export PATH=/data/apps/gcc/11.5/bin:$ISAM_ENV/bin:$PATH
unset PYTHONHOME

export HF_HUB_OFFLINE=1
export INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/InstructSAM}
export INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/InstructSAM/work_dirs/instructsam_stage2_complete_lora}
export PYTHONPATH="$REPO_ROOT/scripts/_env_stubs:$REPO_ROOT:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export COSMOS_SKIP_CUDA_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export INSTRUCTSAM_DECODER_DENSE_SIZE=${INSTRUCTSAM_DECODER_DENSE_SIZE:-32}

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-$DROID_SUCCESS_V21_TAVID_DIR}
export TARGET_FEATURE_DIR_NAME=${TARGET_FEATURE_DIR_NAME:-target_features_instructsam_decoder_dense_stage2_lora}

mkdir -p "$DROID_SUCCESS_V21_TAVID_DIR/$TARGET_FEATURE_DIR_NAME" "$DROID_SUCCESS_V21_TAVID_VAL_DIR/$TARGET_FEATURE_DIR_NAME"

python - <<'PY'
import os
from pathlib import Path
import tokenizers
import transformers

print("python:", os.popen("which python").read().strip())
print("tokenizers:", tokenizers.__version__)
print("transformers:", transformers.__version__, transformers.__file__)
print("instructsam_source:", os.environ["INSTRUCTSAM_SOURCE_ROOT"], Path(os.environ["INSTRUCTSAM_SOURCE_ROOT"]).exists())
print("instructsam_model:", os.environ["INSTRUCTSAM_MODEL_PATH"], Path(os.environ["INSTRUCTSAM_MODEL_PATH"]).exists())
print("decoder_dense_size:", os.environ["INSTRUCTSAM_DECODER_DENSE_SIZE"])
for key in ("DROID_SUCCESS_V21_TAVID_DIR", "DROID_SUCCESS_V21_TAVID_VAL_DIR"):
    dataset_dir = Path(os.environ[key])
    feature_dir = dataset_dir / os.environ["TARGET_FEATURE_DIR_NAME"]
    exclude_path = dataset_dir / "exclude_no_tgt_stems.txt"
    excluded = set(exclude_path.read_text().split()) if exclude_path.exists() else set()
    videos = sorted((dataset_dir / "videos").glob("*.mp4"))
    active = [path for path in videos if path.stem not in excluded]
    print(key, dataset_dir)
    print("videos_active:", len(active), "features_existing:", len(list(feature_dir.glob("*.pt"))), "feature_dir:", feature_dir)
PY

nvidia-smi -L

NPROC=${NPROC:-8}
DATASET_ARGS=(--dataset-dir "$DROID_SUCCESS_V21_TAVID_DIR")
if [ "$DROID_SUCCESS_V21_TAVID_VAL_DIR" != "$DROID_SUCCESS_V21_TAVID_DIR" ]; then
  DATASET_ARGS+=(--dataset-dir "$DROID_SUCCESS_V21_TAVID_VAL_DIR")
fi
torchrun --standalone --nproc_per_node="$NPROC" scripts/precompute_instructsam_target_features.py \
  "${DATASET_ARGS[@]}" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --output-dir-name "$TARGET_FEATURE_DIR_NAME" \
  --feature-mode decoder_dense \
  --expected-feature-dim 256 \
  --combine-mode best \
  --fallback-zero-on-missing-feature \
  --fallback-zero-tokens 1024 \
  --skip-existing \
  --log-every 25

python - <<'PY'
import os
import sys
from pathlib import Path

out_name = os.environ["TARGET_FEATURE_DIR_NAME"]

def validate(dataset_dir: str) -> bool:
    dataset_dir = Path(dataset_dir)
    videos = sorted((dataset_dir / "videos").glob("*.mp4"))
    exclude_path = dataset_dir / "exclude_no_tgt_stems.txt"
    excluded = set(exclude_path.read_text().split()) if exclude_path.exists() else set()
    active = [path for path in videos if path.stem not in excluded]
    missing = [p.stem for p in active if not (dataset_dir / out_name / f"{p.stem}.pt").exists()]
    print(f"validate {dataset_dir} active={len(active)} missing={len(missing)}")
    if missing:
        print("first_missing:", missing[:20])
    return not missing

ok = validate(os.environ["DROID_SUCCESS_V21_TAVID_DIR"])
ok = validate(os.environ["DROID_SUCCESS_V21_TAVID_VAL_DIR"]) and ok
sys.exit(0 if ok else 1)
PY
