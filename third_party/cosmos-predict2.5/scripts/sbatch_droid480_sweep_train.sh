#!/usr/bin/env bash

#SBATCH --job-name=droid480-train
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-droid480-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-droid480-%j.err

set -euo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

bash scripts/run_droid480_sweep_train_remote.sh
