#!/usr/bin/env bash
#SBATCH --job-name=geact-baton-s3
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm-geact-baton-s3-%j.out

set -euo pipefail

GE_ACT_ROOT="${GE_ACT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ ! -f "$GE_ACT_ROOT/main.py" ]]; then
  echo "GE_ACT_ROOT does not contain main.py: $GE_ACT_ROOT" >&2
  exit 2
fi
if [[ -z "${CONDA_ENV:-}" || ! -x "$CONDA_ENV/bin/python" ]]; then
  echo "CONDA_ENV must contain the pinned GE-Act bin/python" >&2
  exit 2
fi
export PYTHON_BIN="$CONDA_ENV/bin/python"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NPROC_PER_NODE=8
CONFIG="${CONFIG:-$GE_ACT_ROOT/configs/ltx_model/libero/action_model_libero_baton_stage3_hdf5.yaml}"

mkdir -p "$GE_ACT_ROOT/logs"
cd "$GE_ACT_ROOT"
bash scripts/train_ltx_baton_stage3.sh "$CONFIG"
