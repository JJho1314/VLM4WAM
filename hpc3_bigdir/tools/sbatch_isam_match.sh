#!/bin/bash
#SBATCH --job-name=isam-match
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:28:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/isam-match-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/isam-match-%j.out
set -uo pipefail
module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true
export REPO_ROOT=/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5
export INSTRUCTSAM_SOURCE_ROOT=/data/user/jhe724/workspace/InstructSAM
export INSTRUCTSAM_MODEL_PATH=/data/user/jhe724/workspace/InstructSAM/work_dirs/instructsam_stage2_complete_lora
export PYTHONPATH="$REPO_ROOT/scripts/_env_stubs:$REPO_ROOT:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 COSMOS_SKIP_CUDA_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false
PY=/data/user/jhe724/.conda/envs/instructsam/bin/python
echo "host=$(hostname) start=$(date)"
$PY /data/user/jhe724/workspace/VLM4WAM/tools/isam_match.py
echo "exit=$? end=$(date)"
