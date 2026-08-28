#!/usr/bin/env bash
#SBATCH --job-name=mask-iou-full
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-mask-iou-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-mask-iou-%j.err
set -uo pipefail
ISAM_ENV=/data/user/jhe724/.conda/envs/instructsam
export PATH=$ISAM_ENV/bin:$PATH
unset PYTHONHOME
python /data/user/jhe724/workspace/VLM4WAM/mg_eval/compare_full_masks.py
echo "compare_exit=$?"
