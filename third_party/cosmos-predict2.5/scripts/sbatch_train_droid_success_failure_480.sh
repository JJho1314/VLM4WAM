#!/usr/bin/env bash
# Run Cosmos 2B post-training on pre-resized DROID success/failure 480x864 data.

#SBATCH --job-name=cosmos-droid480
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-%j.err

set -uo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1

export WANDB_MODE=online
export WANDB_BASE_URL="http://10.12.1.245:8080"
export WANDB_API_KEY="local-37151658708fac20809135dce9e234842db32f97"

export IMAGINAIRE_OUTPUT_ROOT=/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_failure_base_30k_clean_val1000_scratch_480
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

nvidia-smi -L
python -c "import torch; print('cuda count:', torch.cuda.device_count())"

GRAD_ACCUM_ITER=${GRAD_ACCUM_ITER:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_ITER=${MAX_ITER:-10000}
JOB_NAME=${JOB_NAME:-2b_droid_success_failure_base_30k_480_clean_val1000_scratch}

echo "=== TRAIN base 2B on 480x864 droid_success train split + all droid_failure; batch_size=${BATCH_SIZE}; grad_accum=${GRAD_ACCUM_ITER}; max_iter=${MAX_ITER}; job_name=${JOB_NAME} ==="
torchrun --standalone --nproc_per_node=8 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- experiment=predict2_video2world_training_2b_droid_success_failure \
  job.name="$JOB_NAME" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  trainer.grad_accum_iter="$GRAD_ACCUM_ITER" \
  trainer.max_iter="$MAX_ITER" \
  trainer.validation_iter="$MAX_ITER"
status=$?
echo "train_exit=$status"
exit "$status"
