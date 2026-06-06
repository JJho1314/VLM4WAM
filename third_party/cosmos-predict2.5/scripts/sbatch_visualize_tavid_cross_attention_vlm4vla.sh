#!/usr/bin/env bash
# Visualize target-token/feature cross-attention for the isolated VLM4VLA Cosmos run.

#SBATCH --job-name=cosmos-tgtattn-viz
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4VLA/slurm-target-attn-viz-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4VLA/slurm-target-attn-viz-%j.err

set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4VLA/third_party/cosmos-predict2.5}
VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jhe724/workspace/VLM4VLA}
if [ ! -f "$REPO_ROOT/scripts/visualize_tavid_cross_attention.py" ]; then
  echo "Invalid REPO_ROOT=${REPO_ROOT}; visualization script not found." >&2
  exit 2
fi
cd "$REPO_ROOT"

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=${VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/user/jhe724/workspace/weights}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export WANDB_MODE=disabled
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://10.12.1.245:8080}
export WANDB_API_KEY=${WANDB_API_KEY:-local-37151658708fac20809135dce9e234842db32f97}

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2,3,4}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$VLM4VLA_ROOT/outputs/droid_success_v21_instructsam_feature_context_strict_holdout_v3}

BATCH_SIZE=${BATCH_SIZE:-1}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-4}
TAVID_ATTN_QUERY_CHUNK_SIZE=${TAVID_ATTN_QUERY_CHUNK_SIZE:-1024}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context}
JOB_NAME=${JOB_NAME:-2b_droid_success_v21_instructsam_feature_context_strict_v3_49f_s234_bs2accum4_14k_val1000_vlm4vla_from_base_balancedmass_r2}
CHECKPOINT=${CHECKPOINT:-latest}
VIZ_SPLIT=${VIZ_SPLIT:-val}
VIZ_SAMPLES=${VIZ_SAMPLES:-6}
VIZ_MAX_BATCHES=${VIZ_MAX_BATCHES:-80}
VIZ_OUT=${VIZ_OUT:-$VLM4VLA_ROOT/target_attention_vis/${JOB_NAME}_${CHECKPOINT}_${VIZ_SPLIT}}
VIZ_BLOCKS=${VIZ_BLOCKS:-8,12,16,20}
VIZ_SELECTED_BLOCKS=${VIZ_SELECTED_BLOCKS:-8,12,16,20}
VIZ_TOKEN_SOURCE=${VIZ_TOKEN_SOURCE:-config}

mkdir -p "$VLM4VLA_ROOT" "$VIZ_OUT"

echo "date=$(date)"
echo "host=$(hostname)"
echo "repo=$REPO_ROOT"
echo "checkpoint=$CHECKPOINT"
echo "job_name=$JOB_NAME"
echo "output_root=$IMAGINAIRE_OUTPUT_ROOT"
echo "viz_out=$VIZ_OUT"
echo "split=$VIZ_SPLIT samples=$VIZ_SAMPLES max_batches=$VIZ_MAX_BATCHES blocks=$VIZ_BLOCKS selected=$VIZ_SELECTED_BLOCKS token_source=$VIZ_TOKEN_SOURCE"

nvidia-smi -L
python -c "import torch; print('cuda count:', torch.cuda.device_count())"

torchrun --standalone --nproc_per_node=8 -m scripts.visualize_tavid_cross_attention \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint="$CHECKPOINT" \
  --output-dir="$VIZ_OUT" \
  --split="$VIZ_SPLIT" \
  --num-samples="$VIZ_SAMPLES" \
  --max-batches="$VIZ_MAX_BATCHES" \
  --blocks="$VIZ_BLOCKS" \
  --selected-blocks="$VIZ_SELECTED_BLOCKS" \
  --token-source="$VIZ_TOKEN_SOURCE" \
  --sample-label="trained_${CHECKPOINT}" \
  -- experiment="$EXPERIMENT" \
  job.name="$JOB_NAME" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  dataloader_val.batch_size="$BATCH_SIZE" \
  dataloader_val.num_workers="$VAL_NUM_WORKERS" \
  model.config.net.tavid_attn_query_chunk_size="$TAVID_ATTN_QUERY_CHUNK_SIZE"
status=$?
echo "viz_exit=$status"
echo "summary=$VIZ_OUT/cross_attention_visualization_summary.json"
exit "$status"
