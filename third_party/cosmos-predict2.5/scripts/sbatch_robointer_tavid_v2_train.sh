#!/usr/bin/env bash
# TAViD v2: full finetune on RoboInter/LeRobot DROID primary videos with
# CFG-style joint mask + caption dropout (p=0.3), weak attention alignment
# (1 layer, 0.005), lower LR (2^-16) and shorter run (5000 iter) so the base
# autoregressive long-video capability is preserved.

#SBATCH --job-name=cosmos-robo-tavid-v2
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-robointer-tavid-v2-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-robointer-tavid-v2-%j.err

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

export IMAGINAIRE_OUTPUT_ROOT=/data/user/jhe724/workspace/cosmos-predict2.5/outputs/robointer_droid_tavid_v2
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# venv currently has cosmos_cuda 1.4.x but the source merged upstream v1.5.x;
# our patches don't touch cosmos_cuda APIs so the strict version check is
# overridden here. Remove once `uv sync --extra=cuXXX` is run on the cluster.
export COSMOS_SKIP_CUDA_VERSION_CHECK=1

mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

nvidia-smi -L
python -c "import torch; print('cuda count:', torch.cuda.device_count())"

GRAD_ACCUM_ITER=${GRAD_ACCUM_ITER:-1}     # default 8 GPU * 8 micro * 1 accum = global 64
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_ITER=${MAX_ITER:-20140}                # 128866 samples / global 64 = 2014 iter/epoch * 10 epoch
SAVE_ITER=${SAVE_ITER:-2000}
CYCLE_LENGTH=${CYCLE_LENGTH:-30000}
JOB_NAME=${JOB_NAME:-2b_robointer_droid_tavid_v2_10epoch_bs64}

echo "=== TRAIN TAViD v2 (full FT, TAViD-faithful, frame_stride); per_gpu_batch=${BATCH_SIZE}; grad_accum=${GRAD_ACCUM_ITER}; global_batch=$((BATCH_SIZE * 8 * GRAD_ACCUM_ITER)); max_iter=${MAX_ITER}; save_iter=${SAVE_ITER}; cycle=${CYCLE_LENGTH}; job_name=${JOB_NAME} ==="
torchrun --standalone --nproc_per_node=8 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- experiment=predict2_video2world_training_2b_robointer_droid_tavid_v2 \
  job.name="$JOB_NAME" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  trainer.grad_accum_iter="$GRAD_ACCUM_ITER" \
  trainer.max_iter="$MAX_ITER" \
  checkpoint.save_iter="$SAVE_ITER" \
  scheduler.cycle_lengths="[$CYCLE_LENGTH]"
status=$?
echo "train_exit=$status"
exit "$status"
