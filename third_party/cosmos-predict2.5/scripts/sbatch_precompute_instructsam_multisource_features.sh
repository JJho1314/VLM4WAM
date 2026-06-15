#!/usr/bin/env bash
# Precompute fused multi-source InstructSAM target features (mask+detect+vtext ->
# one [L,256] tensor) into target_features_multisource/ for text-free Cosmos.
# Defaults to the lightweight cap200_tasktarget holdout; override the *_DIR vars.

#SBATCH --job-name=precomp-isam-ms
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-precompute-isam-ms-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-precompute-isam-ms-%j.err

set -euo pipefail
# Run from THIS repo (the VLM4WAM copy), not jhe724's separate cosmos project.
# NOTE: Slurm spools the batch script to /var/spool/slurmd, so BASH_SOURCE can't
# locate the repo; use the deploy path (overridable via REPO_ROOT).
REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"
mkdir -p /data/user/jhe724/workspace/VLM4WAM/logs

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

# InstructSAM inference needs the InstructSAM conda env (custom transformers +
# compatible huggingface_hub) -- the cosmos uv venv's newer hf_hub breaks the
# InstructSAM transformers. cosmos_predict2 is imported only for the light bridge
# helpers; a stub cosmos_cuda + COSMOS_SKIP_CUDA_VERSION_CHECK lets that import
# succeed in this env without the CUDA extra. (Training, which DOES need the
# cosmos venv, is a separate script.)
ISAM_ENV=${ISAM_ENV:-/data/user/jhe724/.conda/envs/instructsam}
export PATH=/data/apps/gcc/11.5/bin:$ISAM_ENV/bin:$PATH
unset PYTHONHOME

export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1

export INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/InstructSAM}
export INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/InstructSAM/work_dirs/InstructSAM-2B}
# stub cosmos_cuda dir : this repo (cosmos_predict2) : InstructSAM source.
# Do NOT add the transformers shim -- the conda env has its own transformers.
export PYTHONPATH="$REPO_ROOT/scripts/_env_stubs:$REPO_ROOT:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export COSMOS_SKIP_CUDA_VERSION_CHECK=1

# Lightweight cap200_tasktarget holdout (override as needed).
export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_scene_cap200_tasktarget}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3_scene_cap200_tasktarget}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

OUTPUT_DIR_NAME=${OUTPUT_DIR_NAME:-target_features_multisource}
# NOTE: these budgets MUST match _DROID_SUCCESS_V21_MULTISOURCE_SEGMENTS in
# robointer.py (the DiT's per-source segment embedding + the dataset
# target_feature_max_tokens assume the same [mask, detect, vtext] layout).
MASK_TOKENS=${MASK_TOKENS:-16}
DETECT_TOKENS=${DETECT_TOKENS:-16}
VTEXT_TOKENS=${VTEXT_TOKENS:-32}

# Always precompute train; precompute val only when PRECOMPUTE_VAL=1 (default 1)
# and the val dir exists. With validation disabled for a run, set PRECOMPUTE_VAL=0.
DATASET_DIR_ARGS=(--dataset-dir "$DROID_SUCCESS_V21_TAVID_DIR")
PROCESSED_DIRS=("$DROID_SUCCESS_V21_TAVID_DIR")
mkdir -p "$DROID_SUCCESS_V21_TAVID_DIR/$OUTPUT_DIR_NAME"
if [ "${PRECOMPUTE_VAL:-1}" = "1" ] && [ -d "$DROID_SUCCESS_V21_TAVID_VAL_DIR" ]; then
  DATASET_DIR_ARGS+=(--dataset-dir "$DROID_SUCCESS_V21_TAVID_VAL_DIR")
  PROCESSED_DIRS+=("$DROID_SUCCESS_V21_TAVID_VAL_DIR")
  mkdir -p "$DROID_SUCCESS_V21_TAVID_VAL_DIR/$OUTPUT_DIR_NAME"
fi
export PROCESSED_DIRS_STR="${PROCESSED_DIRS[*]}"
echo "precompute dirs: ${PROCESSED_DIRS[*]}"

python - <<'PY'
import transformers
print("precompute_transformers:", transformers.__version__, transformers.__file__)
import transformers.models.qwen3_vl.video_processing_qwen3_vl  # noqa: F401
from instructsam.models import load_pretrained_model  # noqa: F401
PY

nvidia-smi -L

# LIMIT>0 runs a quick smoke over the first N items (no coverage validation).
# NPROC lets a smoke run on 1 GPU; defaults to 8 for the full run.
NPROC=${NPROC:-8}
torchrun --standalone --nproc_per_node="$NPROC" scripts/precompute_instructsam_multisource_features.py \
  "${DATASET_DIR_ARGS[@]}" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --output-dir-name "$OUTPUT_DIR_NAME" \
  --mask-tokens "$MASK_TOKENS" \
  --detect-tokens "$DETECT_TOKENS" \
  --vtext-tokens "$VTEXT_TOKENS" \
  --out-dim 256 \
  --fallback-zero-on-missing-feature \
  --skip-existing \
  --max-errors 500 \
  ${LIMIT:+--limit "$LIMIT"} \
  --debug-shapes \
  --log-every 25

if [ "${LIMIT:-0}" != "0" ]; then
  echo "LIMIT=${LIMIT} set -> smoke run, skipping coverage validation"
  exit 0
fi

python - <<'PY'
import os
import sys
from pathlib import Path

out_name = os.environ.get("OUTPUT_DIR_NAME", "target_features_multisource")
# Tolerate a tiny fraction of unrecoverable samples (the dataset retries on a
# missing feature at train time) so a handful of bad videos do not block the
# chained training job.
TOLERATE = 0.01


def validate(dataset_dir):
    dataset_dir = Path(dataset_dir)
    videos = sorted((dataset_dir / "videos").glob("*.mp4"))
    exclude_path = dataset_dir / "exclude_no_tgt_stems.txt"
    excluded = set(exclude_path.read_text().split()) if exclude_path.exists() else set()
    active = [path for path in videos if path.stem not in excluded]
    missing = [p.stem for p in active if not (dataset_dir / out_name / f"{p.stem}.pt").exists()]
    frac = (len(missing) / len(active)) if active else 0.0
    print(f"validate dataset={dataset_dir} active={len(active)} missing_features={len(missing)} frac={frac:.4f}")
    if missing:
        print("first_missing:", missing[:20])
    return frac <= TOLERATE


dirs = os.environ.get("PROCESSED_DIRS_STR", os.environ["DROID_SUCCESS_V21_TAVID_DIR"]).split()
ok = all(validate(d) for d in dirs)
sys.exit(0 if ok else 1)
PY
