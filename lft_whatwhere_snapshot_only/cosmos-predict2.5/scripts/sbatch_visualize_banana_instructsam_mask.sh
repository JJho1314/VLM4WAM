#!/usr/bin/env bash
# Re-run InstructSAM on the yellow-banana eval input first frame and save mask overlay.

#SBATCH --job-name=banana-isam-mask
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-banana-isam-mask-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-banana-isam-mask-%j.err

set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
REPO_ROOT=${REPO_ROOT:-$VLM4WAM_ROOT/third_party/cosmos-predict2.5}
DATASET_DIR=${DATASET_DIR:-$VLM4WAM_ROOT/eval_prev_iter2000_full/input_datasets/robointer_74616_banana_prompt_targetaware_dataset}
INPUT_PATH=${INPUT_PATH:-$DATASET_DIR/videos/74616_exterior_image_1_left.mp4}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/outputs/eval_target_context_allblocks_iter1600_banana_20260618_212601}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/instructsam_input_mask_check}
QUERY=${QUERY:-Please segment the yellow banana in the image.}

ISAM_ENV=${ISAM_ENV:-/data/user/jhe724/.conda/envs/instructsam}
INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/InstructSAM}
INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/InstructSAM/work_dirs/instructsam_stage2_complete_lora}

cd "$REPO_ROOT"
module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true
export PATH=/data/apps/gcc/11.5/bin:$ISAM_ENV/bin:$PATH
export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++
export VIRTUAL_ENV=$ISAM_ENV
unset PYTHONHOME
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO_ROOT/scripts/_env_stubs:$REPO_ROOT:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export COSMOS_SKIP_CUDA_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export INSTRUCTSAM_DECODER_DENSE_SIZE=${INSTRUCTSAM_DECODER_DENSE_SIZE:-32}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$OUTPUT_DIR"
python scripts/visualize_one_instructsam_mask.py \
  --input-path "$INPUT_PATH" \
  --query "$QUERY" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --feature-mode decoder_dense \
  --combine-mode best \
  --mask-threshold 0.0
