#!/usr/bin/env bash
#SBATCH --job-name=droid-480p-prep
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --nodelist=ACD1-3
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-prep480-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-prep480-%j.err

set -uo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 2>/dev/null || true

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

export FFMPEG=/data/apps/ffmpeg/7.0.2/ffmpeg
export PY=.venv/bin/python

WORKERS=${WORKERS:-64}
echo "host=$(hostname)"
echo "date=$(date)"
echo "cpus_allowed=$(grep Cpus_allowed_list /proc/self/status)"
echo "workers=${WORKERS}"

WORKERS="$WORKERS" bash scripts/prepare_droid_480p_datasets.sh
