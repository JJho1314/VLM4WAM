#!/usr/bin/env bash
# Precompute dense GT-mask spatial target features with the fine-tuned InstructSAM stage2 LoRA checkpoint.

#SBATCH --job-name=isam-ft-spatial
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jli545/workspace/VLM4VLA/slurm-isam-ft-spatial-%j.out
#SBATCH --error=/data/user/jli545/workspace/VLM4VLA/slurm-isam-ft-spatial-%j.err

set -euo pipefail

VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jli545/workspace/VLM4VLA}
REPO_ROOT=${REPO_ROOT:-$VLM4VLA_ROOT/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

# Use the dedicated InstructSAM env. The Cosmos venv has tokenizers 0.21.x and
# is incompatible with the custom InstructSAM transformers checkout.
ISAM_ENV=${ISAM_ENV:-/data/user/jhe724/.conda/envs/instructsam}
export PATH=/data/apps/gcc/11.5/bin:$ISAM_ENV/bin:$PATH
unset PYTHONHOME

export HF_HUB_OFFLINE=1
export INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/InstructSAM}
# Keep the symlink name containing "lora"; the InstructSAM loader uses the path
# name to enter its LoRA loading branch.
export INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/InstructSAM/work_dirs/instructsam_stage2_complete_lora}
export PYTHONPATH="$REPO_ROOT/scripts/_env_stubs:$REPO_ROOT:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export COSMOS_SKIP_CUDA_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_scene_cap200_tasktarget}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}
export TARGET_FEATURE_DIR_NAME=${TARGET_FEATURE_DIR_NAME:-target_features_gt_mask_spatial64_instructsam_stage2_lora}

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
for key in ("DROID_SUCCESS_V21_TAVID_DIR", "DROID_SUCCESS_V21_TAVID_VAL_DIR"):
    dataset_dir = Path(os.environ[key])
    feature_dir = dataset_dir / os.environ["TARGET_FEATURE_DIR_NAME"]
    print(key, dataset_dir)
    print(
        "videos:", len(list((dataset_dir / "videos").glob("*.mp4"))),
        "masks:", len(list((dataset_dir / "masks").glob("*.npz"))),
        "features_existing:", len(list(feature_dir.glob("*.pt"))),
        "feature_dir:", feature_dir,
    )
PY

nvidia-smi -L

NPROC=${NPROC:-8}
torchrun --standalone --nproc_per_node="$NPROC" scripts/precompute_gt_mask_target_features.py \
  --dataset-dir "$DROID_SUCCESS_V21_TAVID_DIR" \
  --dataset-dir "$DROID_SUCCESS_V21_TAVID_VAL_DIR" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --output-dir-name "$TARGET_FEATURE_DIR_NAME" \
  --expected-feature-dim 256 \
  --max-tokens 64 \
  --mask-frame-policy first \
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
