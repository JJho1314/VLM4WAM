#!/usr/bin/env bash
#SBATCH --job-name=conv-matchg
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-conv-matchg-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-conv-matchg-%j.err
set -uo pipefail
REPO_ROOT=/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5
cd "$REPO_ROOT"
VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export PATH=$VENV/bin:$PATH
export PYTHONPATH="$REPO_ROOT"
NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cublas/lib:$NV_LIB/nccl/lib:${LD_LIBRARY_PATH:-}"
CKPT=/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_match_ground_2000_vlm4wam/cosmos_predict_v2p5/video2world/2b_droid_success_v21_match_ground_cap200_49f_bs2accum8_gbs128_2000/checkpoints/iter_000002000
python scripts/convert_distcp_to_pt.py "$CKPT/model" "$CKPT"
echo "convert_exit=$?"
ls -la "$CKPT" | grep -E "model.*pt"
