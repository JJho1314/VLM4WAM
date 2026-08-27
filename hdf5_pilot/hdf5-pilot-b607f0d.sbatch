#!/bin/bash
#SBATCH --job-name=libero_h5_pilot
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jhe724/junjie/logs/libero_h5_pilot_%j.out
#SBATCH --error=/data/user/jhe724/junjie/logs/libero_h5_pilot_%j.err

set -euo pipefail

CODE_ROOT=/data/user/jhe724/junjie/workspace/VLM4WAM_hdf5_pilot_b607f0d
PYTHON=/data/user/jhe724/.conda/envs/genie_envisioner/bin/python
DATA_ROOT=/data/user/jhe724/junjie/datasets/LIBERO-fastwam
CACHE_ROOT=/data/user/jhe724/junjie/datasets/LIBERO-fastwam-predecoded-rgb

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd "${CODE_ROOT}"

"${PYTHON}" -m ge_act.scripts.convert_libero_fastwam_hdf5 \
  --data-root "${DATA_ROOT}" \
  --predecoded-root "${CACHE_ROOT}" \
  --domains \
    libero_10_no_noops_lerobot \
    libero_goal_no_noops_lerobot \
    libero_object_no_noops_lerobot \
    libero_spatial_no_noops_lerobot \
  --output-root /data/user/jhe724/junjie/datasets/LIBERO-fastwam-hdf5-pilot-none \
  --max-episodes 64 \
  --episodes-per-shard 32 \
  --compression none \
  --resize-microbatch 16

"${PYTHON}" -m ge_act.scripts.convert_libero_fastwam_hdf5 \
  --data-root "${DATA_ROOT}" \
  --predecoded-root "${CACHE_ROOT}" \
  --domains \
    libero_10_no_noops_lerobot \
    libero_goal_no_noops_lerobot \
    libero_object_no_noops_lerobot \
    libero_spatial_no_noops_lerobot \
  --output-root /data/user/jhe724/junjie/datasets/LIBERO-fastwam-hdf5-pilot-lzf \
  --max-episodes 64 \
  --episodes-per-shard 32 \
  --compression lzf \
  --resize-microbatch 16
