#!/bin/bash
#SBATCH --job-name=eval-where-pc
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:28:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/eval-where-pc-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/eval-where-pc-%j.out
set -uo pipefail
module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true
ISAM_ENV=/data/user/jhe724/.conda/envs/instructsam
export PATH=$ISAM_ENV/bin:$PATH
export REPO_ROOT=/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5
export INSTRUCTSAM_SOURCE_ROOT=/data/user/jhe724/workspace/InstructSAM
export INSTRUCTSAM_MODEL_PATH=/data/user/jhe724/workspace/InstructSAM/work_dirs/instructsam_stage2_complete_lora
export PYTHONPATH="$REPO_ROOT/scripts/_env_stubs:$REPO_ROOT:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 COSMOS_SKIP_CUDA_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false
export OUT_DIR_NAME=target_features_where_softlogit_stage2_lora SKIP_EXISTING=0 LIMIT=0
PC=$REPO_ROOT/../../tools/precompute_where_softlogit.py   # resolves to .../VLM4WAM/tools/...
PC=/data/user/jhe724/workspace/VLM4WAM/tools/precompute_where_softlogit.py
BASE=/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets
echo "=== CARROT ==="
DSDIR=$BASE/robointer_74616_yellow_carrot_prompt_targetaware_dataset \
DENSE_DIR_NAME=target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt_s20260613 \
$ISAM_ENV/bin/python $PC
echo "=== BANANA ==="
DSDIR=$BASE/robointer_74616_banana_prompt_targetaware_dataset \
DENSE_DIR_NAME=target_features_instructsam_decoder_dense_stage2_lora_banana_prompt_s20260613 \
$ISAM_ENV/bin/python $PC
echo "exit=$? end=$(date)"
