#!/usr/bin/env bash
#SBATCH --job-name=v21-tavid-prep
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-v21-tavid-prep-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-v21-tavid-prep-%j.err

set -uo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 2>/dev/null || true

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

export PY=.venv/bin/python
export FFMPEG=/data/apps/ffmpeg/7.0.2/ffmpeg
export WORKERS=${WORKERS:-64}
export CLEAN=${CLEAN:-1}

echo "host=$(hostname)"
echo "date=$(date)"
echo "cpus_allowed=$(grep Cpus_allowed_list /proc/self/status)"

bash scripts/prepare_droid_v21_target_aware_hpc.sh
