#!/usr/bin/env bash

#SBATCH --job-name=tavid-gen
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-tavid-gen-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-tavid-gen-%j.err

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
export WANDB_MODE=disabled
export IMAGINAIRE_OUTPUT_ROOT=/data/user/jhe724/workspace/cosmos-predict2.5/outputs/tavid_generation_runs
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

OUT=${OUT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/tavid_generation_runs/final_10k_tavid_style}
CKPT=${CKPT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/robointer_droid_tavid_mask_primary/cosmos_predict_v2p5/video2world/2b_robointer_droid_tavid_mask_primary_10k_bs64/checkpoints/iter_000010000}
NUM_SAMPLES=${NUM_SAMPLES:-8}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
SEED=${SEED:-123}
FPS=${FPS:-8}
NUM_FRAMES=${NUM_FRAMES:-33}

mkdir -p "$OUT"

nvidia-smi -L
python -c "import torch; print('cuda count:', torch.cuda.device_count())"

echo "=== TAViD-style generation ckpt=$CKPT out=$OUT samples=$NUM_SAMPLES frames=$NUM_FRAMES steps=$NUM_STEPS guidance=$GUIDANCE fps=$FPS ==="
torchrun --standalone --nproc_per_node=8 scripts/generate_tavid_mask_samples.py \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint="$CKPT" \
  --output-dir="$OUT" \
  --num-samples="$NUM_SAMPLES" \
  --num-steps="$NUM_STEPS" \
  --guidance="$GUIDANCE" \
  --seed="$SEED" \
  --fps="$FPS" \
  -- experiment=predict2_video2world_training_2b_robointer_droid_tavid_mask \
  dataloader_train.dataset.num_frames="$NUM_FRAMES" \
  dataloader_train.batch_size=1 \
  trainer.grad_accum_iter=1
status=$?
echo "generation_exit=$status"
exit "$status"
