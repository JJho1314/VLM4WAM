#!/usr/bin/env bash
# Run Cosmos 2B baseline post-training on DROID success v2.1 external-camera data.

#SBATCH --job-name=cosmos-v21-base
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-v21-baseline-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-v21-baseline-%j.err

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

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val}
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2,3,4}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}
export IMAGINAIRE_OUTPUT_ROOT=/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_baseline_nomask_noloss_left_right_split
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

nvidia-smi -L
python -c "import torch; print('cuda count:', torch.cuda.device_count())"
python - <<'PY'
import os


def count_active(dataset_dir):
    videos = [f for f in os.listdir(os.path.join(dataset_dir, "videos")) if f.endswith(".mp4")]
    exclude_file = os.path.join(dataset_dir, "exclude_no_tgt_stems.txt")
    excluded = set(open(exclude_file).read().split()) if os.path.exists(exclude_file) else set()
    active = sum(1 for f in videos if os.path.splitext(f)[0] not in excluded)
    return len(videos), len(excluded), active


train_dir = os.environ["DROID_SUCCESS_V21_TAVID_DIR"]
val_dir = os.environ["DROID_SUCCESS_V21_TAVID_VAL_DIR"]
tr_raw, tr_ex, tr_active = count_active(train_dir)
va_raw, va_ex, va_active = count_active(val_dir)
print("train_dataset:", train_dir)
print("train_videos_raw:", tr_raw)
print("train_excluded_no_tgt:", tr_ex)
print("train_videos_active:", tr_active)
print("val_dataset:", val_dir)
print("val_videos_raw:", va_raw)
print("val_excluded_no_tgt:", va_ex)
print("val_videos_active:", va_active)
print("target_masks_used: False")
print("strip_tgt_token: True")
print("num_frames:", os.environ["DROID_SUCCESS_V21_TAVID_NUM_FRAMES"])
print("frame_strides:", os.environ["DROID_SUCCESS_V21_TAVID_FRAME_STRIDES"])
print("frame_start_policy:", os.environ["DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY"])
PY

GRAD_ACCUM_ITER=${GRAD_ACCUM_ITER:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_ITER=${MAX_ITER:-14000}
RUN_VALIDATION=${RUN_VALIDATION:-True}
VALIDATION_ITER=${VALIDATION_ITER:-1000}
MAX_VAL_ITER=${MAX_VAL_ITER:-64}
RUN_VALIDATION_ON_START=${RUN_VALIDATION_ON_START:-False}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_baseline_nomask_noloss}
JOB_NAME=${JOB_NAME:-2b_droid_success_v21_baseline_nomask_noloss_480_lr_split_val1k_49f_s234_actionstart_bs4accum2_14k_val1000}

echo "=== TRAIN DROID success v21 baseline; experiment=${EXPERIMENT}; target_masks_used=False; target_attention_loss_weight=0; per_gpu_batch=${BATCH_SIZE}; grad_accum=${GRAD_ACCUM_ITER}; global_batch=$((BATCH_SIZE * 8 * GRAD_ACCUM_ITER)); max_iter=${MAX_ITER}; validation=${RUN_VALIDATION}; validation_iter=${VALIDATION_ITER}; max_val_iter=${MAX_VAL_ITER}; job_name=${JOB_NAME} ==="
torchrun --standalone --nproc_per_node=8 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- experiment="$EXPERIMENT" \
  job.name="$JOB_NAME" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  trainer.grad_accum_iter="$GRAD_ACCUM_ITER" \
  trainer.max_iter="$MAX_ITER" \
  trainer.run_validation="$RUN_VALIDATION" \
  trainer.validation_iter="$VALIDATION_ITER" \
  trainer.max_val_iter="$MAX_VAL_ITER" \
  trainer.run_validation_on_start="$RUN_VALIDATION_ON_START"
status=$?
echo "train_exit=$status"
exit "$status"
