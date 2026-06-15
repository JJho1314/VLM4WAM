#!/usr/bin/env bash
# Precompute RAW [SEG] hidden-state features (feature_mode=raw_seg, 2048-d) into
# target_features_rawseg/ for the dense-spatial rawseg ablation. Uses the SAME
# original InstructSAM-2B as the 256-d baseline so the only variable is the
# feature type. InstructSAM conda env (same recipe as the multisource precompute).

#SBATCH --job-name=precomp-rawseg
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-precomp-rawseg-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-precomp-rawseg-%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"
mkdir -p /data/user/jhe724/workspace/VLM4WAM/logs

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

ISAM_ENV=${ISAM_ENV:-/data/user/jhe724/.conda/envs/instructsam}
export PATH=/data/apps/gcc/11.5/bin:$ISAM_ENV/bin:$PATH
unset PYTHONHOME
export HF_HUB_OFFLINE=1
export INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/InstructSAM}
# Original InstructSAM-2B (same extractor as the 256-d baseline features).
export INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/InstructSAM/work_dirs/InstructSAM-2B}
export PYTHONPATH="$REPO_ROOT/scripts/_env_stubs:$REPO_ROOT:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export COSMOS_SKIP_CUDA_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_scene_cap200_tasktarget}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}

OUTPUT_DIR_NAME=${OUTPUT_DIR_NAME:-target_features_rawseg}
mkdir -p "$DROID_SUCCESS_V21_TAVID_DIR/$OUTPUT_DIR_NAME" "$DROID_SUCCESS_V21_TAVID_VAL_DIR/$OUTPUT_DIR_NAME"

nvidia-smi -L
torchrun --standalone --nproc_per_node=8 scripts/precompute_instructsam_target_features.py \
  --dataset-dir "$DROID_SUCCESS_V21_TAVID_DIR" \
  --dataset-dir "$DROID_SUCCESS_V21_TAVID_VAL_DIR" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --output-dir-name "$OUTPUT_DIR_NAME" \
  --feature-mode raw_seg \
  --expected-feature-dim 2048 \
  --combine-mode best \
  --fallback-zero-on-missing-feature \
  --fallback-zero-tokens 64 \
  --skip-existing \
  --max-errors 500 \
  --log-every 25

python - <<'PY'
import os
import sys
from pathlib import Path

out_name = os.environ.get("OUTPUT_DIR_NAME", "target_features_rawseg")
TOLERATE = 0.01


def validate(dataset_dir):
    dataset_dir = Path(dataset_dir)
    videos = sorted((dataset_dir / "videos").glob("*.mp4"))
    exclude = dataset_dir / "exclude_no_tgt_stems.txt"
    excluded = set(exclude.read_text().split()) if exclude.exists() else set()
    active = [p for p in videos if p.stem not in excluded]
    missing = [p.stem for p in active if not (dataset_dir / out_name / f"{p.stem}.pt").exists()]
    frac = (len(missing) / len(active)) if active else 0.0
    print(f"validate {dataset_dir} active={len(active)} missing={len(missing)} frac={frac:.4f}")
    return frac <= TOLERATE


ok = validate(os.environ["DROID_SUCCESS_V21_TAVID_DIR"])
ok = validate(os.environ["DROID_SUCCESS_V21_TAVID_VAL_DIR"]) and ok
sys.exit(0 if ok else 1)
PY
