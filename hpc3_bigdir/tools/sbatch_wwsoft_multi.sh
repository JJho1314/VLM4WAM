#!/bin/bash
#SBATCH --job-name=wwsoft-multi
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/wwsoft-multi-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/wwsoft-multi-%j.err
set -uo pipefail
REPO_ROOT=/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5
cd "$REPO_ROOT" || exit 2
module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true
VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-oss:${PYTHONPATH:-}"
export CC=/data/apps/gcc/11.5/bin/gcc CXX=/data/apps/gcc/11.5/bin/g++
NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_VERBOSITY=error WANDB_MODE=disabled COSMOS_SKIP_CUDA_VERSION_CHECK=1
export DROID_SUCCESS_V21_TAVID_DIR=/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half
export DROID_SUCCESS_V21_TAVID_VAL_DIR=$DROID_SUCCESS_V21_TAVID_DIR
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=49 DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=1,2,3 DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=range_start

CKPT=/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_v21_what_where_softlogit_320x576_49f_s123_vlm4wam/cosmos_predict_v2p5/video2world/2b_what_where_softlogit_iou50_320x576_49f_s123_bs4accum4_gbs128_3000/checkpoints/iter_000003000
OUT=/data/user/jhe724/workspace/VLM4WAM/feature_guidance_analysis/what_where_softlogit_iter3000_multi12
mkdir -p "$OUT"
echo "host=$(hostname) start=$(date)"
torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --num-samples 12 --num-steps 35 --guidance 3.0 --seed 20260613 --fps 8 \
  --max-batches 12 --standalone-only --reuse-encoded-latent \
  --offload-denoiser-during-vae --offload-denoiser-before-decode \
  --allow-empty-target-mask --remove-target-mask \
  --target-feature-mode keep \
  -- experiment=predict2_video2world_training_2b_droid_success_v21_what_where_softlogit \
  job.name=2b_what_where_softlogit_iou50_320x576_49f_s123_bs4accum4_gbs128_3000 \
  dataloader_train.batch_size=1 dataloader_train.num_workers=2 dataloader_train.drop_last=False \
  dataloader_train.dataset.video_size="[320,576]" dataloader_train.sampler.dataset.video_size="[320,576]" \
  dataloader_train.dataset.caption_dropout_prob=0.0 dataloader_train.sampler.dataset.caption_dropout_prob=0.0 \
  dataloader_train.dataset.target_mask_dropout_prob=0.0 dataloader_train.sampler.dataset.target_mask_dropout_prob=0.0 \
  trainer.grad_accum_iter=1 trainer.run_validation=False \
  model.config.net.tavid_attn_query_chunk_size=1024
echo "exit=$? end=$(date)"
