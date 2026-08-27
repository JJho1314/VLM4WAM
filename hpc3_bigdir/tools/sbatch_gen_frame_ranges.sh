#!/bin/bash
#SBATCH --job-name=gen-frameranges
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:28:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/gen-frameranges-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/gen-frameranges-%j.out
export DSDIR=/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half
export WORKERS=12
PY=/data/user/jhe724/.conda/envs/fastwam/bin/python
echo "host=$(hostname) start=$(date)"
$PY /data/user/jhe724/workspace/VLM4WAM/tools/gen_frame_ranges.py run
echo "exit=$? end=$(date)"
