#!/usr/bin/env bash
#SBATCH --job-name=v21-tavid-viz
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-v21-tavid-viz-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-v21-tavid-viz-%j.err

set -uo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 2>/dev/null || true

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

PY=${PY:-.venv/bin/python}
DATASET_DIR=${DATASET_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864}
VIZ_OUT=${VIZ_OUT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_target_aware_viz_more}
VIZ_SAMPLES=${VIZ_SAMPLES:-96}
VIZ_FRAMES=${VIZ_FRAMES:-6}
VIZ_SEED=${VIZ_SEED:-20260521}

echo "date=$(date)"
echo "host=$(hostname)"
echo "dataset=$DATASET_DIR"
echo "viz_out=$VIZ_OUT"
echo "samples=$VIZ_SAMPLES frames=$VIZ_FRAMES seed=$VIZ_SEED"

"$PY" scripts/visualize_target_masks.py \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$VIZ_OUT" \
  --num-samples "$VIZ_SAMPLES" \
  --frames-per-sample "$VIZ_FRAMES" \
  --seed "$VIZ_SEED"

echo "viz_count=$(find "$VIZ_OUT" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
echo "overview=$VIZ_OUT/overview.jpg"
