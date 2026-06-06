#!/usr/bin/env bash
#SBATCH --job-name=droid-fail-reenc
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-%j.err

set -euo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=$VENV/bin:$PATH
export FFMPEG=/data/apps/ffmpeg/7.0.2/ffmpeg

.venv/bin/python scripts/reencode_video_dataset.py \
  --src /data/user/jhe724/workspace/datasets/droid_failure_left_all \
  --out /data/user/jhe724/workspace/datasets/droid_failure_left_all_clean \
  --workers 8 \
  --timeout-sec 300 \
  --status-csv /data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_failure_reencode_status.csv
