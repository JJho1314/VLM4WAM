#!/usr/bin/env bash
# LingBot-native current + one-future planner with four independent modality/time
# task-token groups on HPC3 (jhe724).
# Global batch = 8 GPUs * 16 samples/GPU * 1 accumulation = 128.
#
# Smoke:
#   sbatch --export=ALL,RUN_KIND=smoke,MAX_STEPS=2,SAVE_STEPS=2 \
#     scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_hpc3.sbatch
# Formal:
#   sbatch --export=ALL,RUN_KIND=formal,MAX_STEPS=12000,SAVE_STEPS=1000 \
#     scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_hpc3.sbatch
#SBATCH -J vlmp_k1_indq_g128
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 2-00:00:00
#SBATCH -o /data/user/jhe724/workspace/VLM4WAM_lingbot_k1_current_20260712/logs/vlm_planner_k1_indq_%j.log

set -euo pipefail

WORKSPACE=/data/user/jhe724/workspace
REPO_ROOT=$WORKSPACE/VLM4WAM_lingbot_k1_current_20260712
DATA_ROOT=$WORKSPACE/datasets/LIBERO-fastwam
TEXT_CACHE=$WORKSPACE/datasets/libero_qwen
NORM_STATS=$WORKSPACE/datasets/LIBERO-fastwam_meta/dataset_stats.json
WEIGHTS=$WORKSPACE/weights

RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=8
NUM_TASK_TOKENS=64
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_STEPS=${MAX_STEPS:-12000}
SAVE_STEPS=${SAVE_STEPS:-1000}
LOG_STEPS=${LOG_STEPS:-10}
GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

if [[ "$GLOBAL_BATCH_SIZE" -ne 128 ]]; then
  echo "ERROR: expected global batch 128, got $GLOBAL_BATCH_SIZE" >&2
  exit 2
fi

for required_path in \
  "$REPO_ROOT/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1.sh" \
  "$REPO_ROOT/third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" \
  "$DATA_ROOT/libero_object_no_noops_lerobot" \
  "$DATA_ROOT/libero_goal_no_noops_lerobot" \
  "$DATA_ROOT/libero_10_no_noops_lerobot" \
  "$TEXT_CACHE" \
  "$NORM_STATS" \
  "$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  "$WEIGHTS/lingbot_align_heads_warmstart/model.safetensors.index.json" \
  "$WEIGHTS/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
  "$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  "$WEIGHTS/moge-2-vitb-normal/model.pt"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: missing required path: $required_path" >&2
    exit 2
  fi
done

if [[ "$RUN_KIND" == "smoke" ]]; then
  OUTPUT_DIR=$REPO_ROOT/outputs/smoke_k1_independent_queries_q64_b${BATCH_SIZE}_a${GRAD_ACCUM}_job${SLURM_JOB_ID}
else
  OUTPUT_DIR=$REPO_ROOT/outputs/qwen3vl4b_lingbot_independent_queries_q64_fastwam_k1_8gpu_b${BATCH_SIZE}_a${GRAD_ACCUM}_gbs128_${MAX_STEPS}step
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PLANNER_WANDB=0
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TRITON_CACHE_DIR=$WORKSPACE/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$WORKSPACE/.cache/inductor
export OMP_NUM_THREADS=4
export INDEPENDENT_MODALITY_TASK_TOKENS=1
mkdir -p "$REPO_ROOT/logs" "$OUTPUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$REPO_ROOT"
echo "[slurm] job_id=$SLURM_JOB_ID host=$(hostname) run_kind=$RUN_KIND"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "[batch] gpu=$NUM_GPUS per_gpu=$BATCH_SIZE accum=$GRAD_ACCUM global=$GLOBAL_BATCH_SIZE"
echo "[train] steps=$MAX_STEPS lr=3e-5 head_lr=3e-4 warmup=1000 cosine bf16 full_ft gradient_checkpointing=off"
echo "[contract] frames=9 offset=8 four_independent_groups=4x64 latent_len=256 output_tokens_per_feature=256"
echo "[loss] current_dino=0.004 future_dino=0.004 current_depth=0.004 future_depth=0.004"
echo "[output] $OUTPUT_DIR"

NUM_GPUS=$NUM_GPUS \
NUM_TASK_TOKENS=$NUM_TASK_TOKENS \
BATCH_SIZE=$BATCH_SIZE \
GRAD_ACCUM=$GRAD_ACCUM \
MAX_STEPS=$MAX_STEPS \
SAVE_STEPS=$SAVE_STEPS \
LOG_STEPS=$LOG_STEPS \
FULL_FINETUNE=1 \
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
bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1.sh
