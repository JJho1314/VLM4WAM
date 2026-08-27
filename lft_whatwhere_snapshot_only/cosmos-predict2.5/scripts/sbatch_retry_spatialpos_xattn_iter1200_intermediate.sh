#!/bin/bash
#SBATCH --job-name=mgv3ctx1200int
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-mgv3ctx1200int-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-mgv3ctx1200int-%j.err

set -euo pipefail

REPO=${REPO:-/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5}
VENV=${VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
RUN_ROOT=${RUN_ROOT:-/data/user/jhe724/workspace/VLM4WAM/feature_guidance_analysis/spatialpos_xattn_iter1200_yellow_carrot_20260619_153151}
CHECKPOINT=${CHECKPOINT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_v21_match_ground_v3_target_context_spatialpos_crossattn_49f_s123_vlm4wam/cosmos_predict_v2p5/video2world/2b_mgv3_target_context_spatialpos_xattn_iou50_49f_s123_bs1accum16_gbs128_2000/checkpoints/iter_000001200}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_match_ground_v3_target_context_spatialpos_crossattn}
JOB_NAME=${JOB_NAME:-2b_mgv3_target_context_spatialpos_xattn_iou50_49f_s123_bs1accum16_gbs128_2000}
CARROT_DS=${CARROT_DS:-/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets/robointer_74616_yellow_carrot_prompt_targetaware_dataset}
CARROT_RAW=${CARROT_RAW:-/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets/robointer_74616_yellow_carrot_prompt_targetaware_dataset/target_features_rawseg_ft/74616_exterior_image_1_left.pt}
CARROT_DENSE=${CARROT_DENSE:-/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets/robointer_74616_yellow_carrot_prompt_targetaware_dataset/target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt_s20260613/74616_exterior_image_1_left.pt}
BANANA_RAW=${BANANA_RAW:-/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets/robointer_74616_banana_prompt_targetaware_dataset/target_features_rawseg_stage2_lora_banana_prompt_s20260613/74616_exterior_image_1_left.pt}
BANANA_DENSE=${BANANA_DENSE:-/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets/robointer_74616_banana_prompt_targetaware_dataset/target_features_instructsam_decoder_dense_stage2_lora_banana_prompt_s20260613/74616_exterior_image_1_left.pt}
SEED=${SEED:-20260613}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
TAVID_ATTN_QUERY_CHUNK_SIZE=${TAVID_ATTN_QUERY_CHUNK_SIZE:-1024}

cd "$REPO"
export VIRTUAL_ENV="$VENV"
export PATH="/data/apps/gcc/11.5/bin:$VENV/bin:$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
NV_LIB="$VENV/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_${USER}_cosmos_intermediate}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_${USER}_cosmos_intermediate}"
export COSMOS_CHECKPOINTS_DIR="${COSMOS_CHECKPOINTS_DIR:-/data/user/jhe724/workspace/weights}"

mkdir -p "$RUN_ROOT/intermediate" "$RUN_ROOT/logs"

torchrun --standalone --nproc_per_node=1 -m scripts.analyze_vlm_feature_guidance_intermediate \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint="$CHECKPOINT" \
  --output-dir="$RUN_ROOT/intermediate" \
  --split=train \
  --num-samples=1 \
  --max-batches=1 \
  --num-conditional-frames=1 \
  --seed="$SEED" \
  --blocks=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
  --selected-blocks=4,8,12,16,20,24 \
  --variants keep zero drop wrong \
  --wrong-target-feature-path="$BANANA_RAW" \
  --wrong-target-dense-feature-path="$BANANA_DENSE" \
  --token-source=feature \
  -- experiment="$EXPERIMENT" \
  job.name="$JOB_NAME" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  dataloader_train.num_workers="$NUM_WORKERS" \
  dataloader_train.drop_last=False \
  dataloader_train.dataset.dataset_dir="$CARROT_DS" \
  dataloader_train.sampler.dataset.dataset_dir="$CARROT_DS" \
  dataloader_train.dataset.target_mask_dir=auto \
  dataloader_train.dataset.target_feature_dir="$(dirname "$CARROT_RAW")" \
  dataloader_train.dataset.target_feature_default_to_zero=False \
  dataloader_train.dataset.target_feature_dim=2048 \
  dataloader_train.dataset.target_feature_max_tokens=16 \
  dataloader_train.dataset.target_dense_feature_dir="$(dirname "$CARROT_DENSE")" \
  dataloader_train.dataset.target_dense_feature_default_to_zero=False \
  dataloader_train.dataset.target_dense_feature_dim=256 \
  dataloader_train.dataset.target_dense_feature_max_tokens=1024 \
  model.config.net.tavid_attn_query_chunk_size="$TAVID_ATTN_QUERY_CHUNK_SIZE" \
  2>&1 | tee "$RUN_ROOT/logs/07_intermediate_guidance_retry.log"
