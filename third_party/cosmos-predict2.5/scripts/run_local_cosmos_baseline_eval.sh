#!/usr/bin/env bash
set -euo pipefail

cd /data/LFT-W02_data/junjie/cosmos-predict2.5

VENV=${VENV:-/data/LFT-W02_data/junjie/cosmos-predict2.5/.venv}
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONHOME

NV_LIB="$VENV/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/LFT-W02_data/junjie/weights}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/LFT-W02_data/junjie/datasets/droid_success_v21_target_aware_left_right_480x864_val}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-$DROID_SUCCESS_V21_TAVID_DIR}
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-2,3,4}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

CKPT=${CKPT:-outputs/droid_success_v21_baseline_nomask_noloss_left_right_split/cosmos_predict_v2p5/video2world/2b_droid_success_v21_baseline_nomask_noloss_480_lr_split_val1k_49f_s234_actionstart_bs4accum2_14k_val1000/checkpoints/iter_000014000}
OUT=${OUT:-outputs/tavid_generation_runs/v21_baseline_14k_val4_49f_35step}
NUM_SAMPLES=${NUM_SAMPLES:-4}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
SEED=${SEED:-20260528}
FPS=${FPS:-8}
MAX_BATCHES=${MAX_BATCHES:-80}
SKIP_SAMPLES=${SKIP_SAMPLES:-0}
SAMPLE_INDEX_OFFSET=${SAMPLE_INDEX_OFFSET:-$SKIP_SAMPLES}
STANDALONE_ONLY=${STANDALONE_ONLY:-0}

mkdir -p "$OUT"

nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
printf 'checkpoint=%s\noutput=%s\ndataset=%s\n' "$CKPT" "$OUT" "$DROID_SUCCESS_V21_TAVID_DIR"

EXTRA_SAMPLE_ARGS=()
if [[ "$STANDALONE_ONLY" == "1" || "$STANDALONE_ONLY" == "true" ]]; then
  EXTRA_SAMPLE_ARGS+=(--standalone-only)
fi

torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --num-samples "$NUM_SAMPLES" \
  --skip-samples "$SKIP_SAMPLES" \
  --sample-index-offset "$SAMPLE_INDEX_OFFSET" \
  --num-steps "$NUM_STEPS" \
  --guidance "$GUIDANCE" \
  --seed "$SEED" \
  --fps "$FPS" \
  --max-batches "$MAX_BATCHES" \
  "${EXTRA_SAMPLE_ARGS[@]}" \
  -- experiment=predict2_video2world_training_2b_droid_success_v21_baseline_nomask_noloss \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2 \
  dataloader_train.dataset.target_mask_dir=auto \
  dataloader_train.dataset.target_mask_default_to_zero=False \
  dataloader_train.dataset.strip_tgt_token=True \
  trainer.grad_accum_iter=1 \
  trainer.run_validation=False
