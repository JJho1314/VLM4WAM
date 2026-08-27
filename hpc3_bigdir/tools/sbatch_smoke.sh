#!/bin/bash
#SBATCH --job-name=smoke-softlogit
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/smoke-softlogit-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/smoke-softlogit-%j.out
set -uo pipefail
module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true
REPO_ROOT=/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5
VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export COSMOS_SKIP_CUDA_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export DROID_SUCCESS_V21_TAVID_DIR=/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half
export DROID_SUCCESS_V21_TAVID_VAL_DIR=$DROID_SUCCESS_V21_TAVID_DIR
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=49 DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=1,2,3 DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=range_start
cd $REPO_ROOT
echo "host=$(hostname) start=$(date)"
$VENV/bin/python /data/user/jhe724/workspace/VLM4WAM/tools/smoke_softlogit.py
echo "exit=$? end=$(date)"
