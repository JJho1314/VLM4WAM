#!/usr/bin/env bash
# Visualize feature-token attention for the mask-free all-block context run.
# The job waits for a checkpoint from the active training run, then exports
# overlay grids, temporal attention-mass strips, raw attention tensors, and a
# JSON summary.

#SBATCH --job-name=mgv3ctx-viz
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-mgv3ctx-viz-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-mgv3ctx-viz-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
REPO_ROOT=${REPO_ROOT:-$VLM4WAM_ROOT/third_party/cosmos-predict2.5}
if [ ! -f "$REPO_ROOT/scripts/visualize_tavid_cross_attention.py" ]; then
  echo "Invalid REPO_ROOT=${REPO_ROOT}; visualization script not found." >&2
  exit 2
fi
cd "$REPO_ROOT"

mkdir -p "$VLM4WAM_ROOT/logs"

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

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

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-1,2,3}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_v21_match_ground_v3_target_context_allblocks_49f_s123_vlm4wam_fix2}

BATCH_SIZE=${BATCH_SIZE:-1}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-4}
TAVID_ATTN_QUERY_CHUNK_SIZE=${TAVID_ATTN_QUERY_CHUNK_SIZE:-1024}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_match_ground_v3_target_context_allblocks}
JOB_NAME=${JOB_NAME:-2b_mgv3_target_context_allblocks_iou50_49f_s123_bs1accum16_gbs128_1600_fix2}
CHECKPOINT=${CHECKPOINT:-latest}
VIZ_SPLIT=${VIZ_SPLIT:-val}
VIZ_SAMPLES=${VIZ_SAMPLES:-8}
VIZ_MAX_BATCHES=${VIZ_MAX_BATCHES:-160}
VIZ_BLOCKS=${VIZ_BLOCKS:-0,4,8,12,16,20,24,27}
VIZ_SELECTED_BLOCKS=${VIZ_SELECTED_BLOCKS:-8,12,16,20}
VIZ_TOKEN_SOURCE=${VIZ_TOKEN_SOURCE:-feature}
VIZ_OUT=${VIZ_OUT:-$VLM4WAM_ROOT/attention_vis/${JOB_NAME}_${CHECKPOINT}_${VIZ_SPLIT}_feature_tokens}
WAIT_FOR_CHECKPOINT_SECONDS=${WAIT_FOR_CHECKPOINT_SECONDS:-10800}
WAIT_POLL_SECONDS=${WAIT_POLL_SECONDS:-120}

RUN_DIR="$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/video2world/$JOB_NAME"
LATEST_FILE="$RUN_DIR/checkpoints/latest_checkpoint.txt"

if [ "$CHECKPOINT" = "latest" ]; then
  waited=0
  while [ ! -s "$LATEST_FILE" ]; do
    if [ "$waited" -ge "$WAIT_FOR_CHECKPOINT_SECONDS" ]; then
      echo "Timed out waiting for $LATEST_FILE" >&2
      exit 3
    fi
    echo "Waiting for checkpoint: $LATEST_FILE (${waited}s/${WAIT_FOR_CHECKPOINT_SECONDS}s)"
    sleep "$WAIT_POLL_SECONDS"
    waited=$((waited + WAIT_POLL_SECONDS))
  done
fi

mkdir -p "$VIZ_OUT"

echo "date=$(date)"
echo "host=$(hostname)"
echo "repo=$REPO_ROOT"
echo "checkpoint=$CHECKPOINT"
echo "latest_file=$LATEST_FILE"
if [ -f "$LATEST_FILE" ]; then
  echo "latest_checkpoint=$(cat "$LATEST_FILE")"
fi
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
  --sample-label="target_context_allblocks_${CHECKPOINT}" \
  --save-debug-pack \
  -- experiment="$EXPERIMENT" \
  job.name="$JOB_NAME" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  dataloader_val.batch_size="$BATCH_SIZE" \
  dataloader_val.num_workers="$VAL_NUM_WORKERS" \
  model.config.net.tavid_attn_query_chunk_size="$TAVID_ATTN_QUERY_CHUNK_SIZE"
status=$?
echo "viz_exit=$status"
echo "summary=$VIZ_OUT/cross_attention_visualization_summary.json"
echo "debug_pack=$VIZ_OUT/attention_debug_pack"
exit "$status"
