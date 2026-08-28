#!/usr/bin/env bash
# FastWAM-aligned 4B VLM planner training on one HPC3 8-GPU node.
#
# The batch decomposition is supplied at submission time, but this launcher
# refuses to run unless the effective global batch is exactly 128.
#
# Smoke (preferred throughput setting):
#   sbatch --export=ALL,RUN_KIND=smoke,BATCH_SIZE=4,GRAD_ACCUM=4,MAX_STEPS=2,SAVE_STEPS=2 \
#     scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4_hpc3.sbatch
# Formal:
#   sbatch --export=ALL,RUN_KIND=formal,BATCH_SIZE=4,GRAD_ACCUM=4,MAX_STEPS=12000 \
#     scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4_hpc3.sbatch
#SBATCH -J vlmp_k4_g128
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 2-00:00:00
#SBATCH -o /data/user/jhe724/workspace/VLM4WAM_k4_gbs128_20260711/logs/vlm_planner_k4_gbs128_%j.log

set -euo pipefail

WORKSPACE=/data/user/jhe724/workspace
REPO_ROOT=$WORKSPACE/VLM4WAM_k4_gbs128_20260711
DATA_ROOT=$WORKSPACE/datasets/LIBERO-fastwam
TEXT_CACHE=$WORKSPACE/datasets/libero_qwen
NORM_STATS=$WORKSPACE/datasets/LIBERO-fastwam_meta/dataset_stats.json
WEIGHTS=$WORKSPACE/weights

RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_STEPS=${MAX_STEPS:-12000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOG_STEPS=${LOG_STEPS:-10}
GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

if [[ "$NUM_GPUS" -ne 8 ]]; then
  echo "ERROR: this launcher requires exactly 8 GPUs, got $NUM_GPUS" >&2
  exit 2
fi
if [[ "$GLOBAL_BATCH_SIZE" -ne 128 ]]; then
  echo "ERROR: expected global batch 128, got $GLOBAL_BATCH_SIZE" >&2
  exit 2
fi

for required_path in \
  "$REPO_ROOT/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4.sh" \
  "$REPO_ROOT/third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" \
  "$DATA_ROOT/libero_object_no_noops_lerobot" \
  "$DATA_ROOT/libero_goal_no_noops_lerobot" \
  "$DATA_ROOT/libero_10_no_noops_lerobot" \
  "$TEXT_CACHE" \
  "$NORM_STATS" \
  "$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  "$WEIGHTS/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
  "$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  "$WEIGHTS/moge-2-vitb-normal/model.pt"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: missing required path: $required_path" >&2
    exit 2
  fi
done

if [[ "$RUN_KIND" == "smoke" ]]; then
  OUTPUT_DIR=$REPO_ROOT/outputs/smoke_k4_b${BATCH_SIZE}_a${GRAD_ACCUM}_gbs128_job${SLURM_JOB_ID}
else
  OUTPUT_DIR=$REPO_ROOT/outputs/qwen3vl4b_lingbot_dino_depth_fastwam_k4_fullft_b${BATCH_SIZE}_a${GRAD_ACCUM}_gbs128_${MAX_STEPS}step
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TRITON_CACHE_DIR=$WORKSPACE/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$WORKSPACE/.cache/inductor
export OMP_NUM_THREADS=4
mkdir -p "$REPO_ROOT/logs" "$OUTPUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$REPO_ROOT"
echo "[slurm] job_id=$SLURM_JOB_ID host=$(hostname) run_kind=$RUN_KIND"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "[batch] num_gpus=$NUM_GPUS batch_per_gpu=$BATCH_SIZE grad_accum=$GRAD_ACCUM global_batch=$GLOBAL_BATCH_SIZE"
echo "[train] max_steps=$MAX_STEPS lr=3e-5 head_lr=3e-4 warmup_steps=1000 full_finetune=1 dtype=bf16"
echo "[contract] sequence_length=9 keyframes=4 offsets=2,4,6,8 queries=32_shared+32_private_per_branch total_unique_queries=384"
echo "[output] $OUTPUT_DIR"

NUM_GPUS=$NUM_GPUS \
BATCH_SIZE=$BATCH_SIZE \
GRAD_ACCUM=$GRAD_ACCUM \
MAX_STEPS=$MAX_STEPS \
SAVE_STEPS=$SAVE_STEPS \
LOG_STEPS=$LOG_STEPS \
FULL_FINETUNE=1 \
USE_DEPTH=1 \
NUM_WORKERS=4 \
LR=3e-5 \
HEAD_LR=3e-4 \
WARMUP_STEPS=1000 \
PY=$HOME/.conda/envs/starVLA/bin/python \
WEIGHTS=$WEIGHTS \
MODEL_PATH=$WEIGHTS/Qwen3-VL-4B-lingbot-vlm \
LINGBOT_6B=$WEIGHTS/lingbot-vla-v2-6b \
HEAD_WARMSTART_CKPT=$WEIGHTS/lingbot_align_heads_warmstart \
DEPTH_MOGE_PATH=$WEIGHTS/moge-2-vitb-normal/model.pt \
DEPTH_MORGBD_PATH=$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt \
LINGBOT_SRC_ROOT=$WORKSPACE/lingbot-vla-v2 \
UTILS3D_MOGE_PATH=$WORKSPACE/py_deps/utils3d_moge \
FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
FASTWAM_DATASET_DIRS=$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot \
FASTWAM_TEXT_EMBEDDING_CACHE_DIR=$TEXT_CACHE \
FASTWAM_PRETRAINED_NORM_STATS=$NORM_STATS \
OUTPUT_DIR=$OUTPUT_DIR \
bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4.sh
