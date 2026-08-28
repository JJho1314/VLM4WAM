#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/nas/junjie
REPO_ROOT=$ROOT/code/VLM4WAM_k1_fastwam_20260712
DATA_ROOT=$ROOT/data/LIBERO-fastwam
WEIGHTS=$ROOT/weights
PY=${PY:-$ROOT/conda_envs/vlm4wam/bin/python}

RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=8
NUM_TASK_TOKENS=64
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_STEPS=${MAX_STEPS:-12000}
SAVE_STEPS=${SAVE_STEPS:-1000}
GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE * GRAD_ACCUM))

if [[ "$GLOBAL_BATCH_SIZE" -ne 128 ]]; then
  echo "ERROR: expected global batch 128, got $GLOBAL_BATCH_SIZE" >&2
  exit 2
fi

for path in \
  "$PY" \
  "$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  "$WEIGHTS/lingbot_align_heads_warmstart/model.safetensors.index.json" \
  "$WEIGHTS/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
  "$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  "$WEIGHTS/moge-2-vitb-normal/model.pt" \
  "$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" \
  "$ROOT/data/libero_qwen" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" \
  "$DATA_ROOT/libero_object_no_noops_lerobot" \
  "$DATA_ROOT/libero_goal_no_noops_lerobot" \
  "$DATA_ROOT/libero_10_no_noops_lerobot"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 2
  fi
done

if [[ "$RUN_KIND" == "smoke" ]]; then
  OUTPUT_DIR=$REPO_ROOT/outputs/smoke_k1_independent_queries_q64_8gpu_b${BATCH_SIZE}a${GRAD_ACCUM}
else
  OUTPUT_DIR=$REPO_ROOT/outputs/qwen3vl4b_lingbot_independent_queries_q64_fastwam_k1_8gpu_b4a4_gbs128_12k
fi
mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs" "$ROOT/cache/triton" "$ROOT/cache/inductor"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PLANNER_WANDB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor
export OMP_NUM_THREADS=4
export INDEPENDENT_MODALITY_TASK_TOKENS=1

cd "$REPO_ROOT"
echo "[pod] run=$RUN_KIND gpu=8 batch=$BATCH_SIZE accum=$GRAD_ACCUM global=$GLOBAL_BATCH_SIZE"
echo "[contract] frames=9 offset=8 four_independent_groups=4x64 latent_len=256 output_tokens_per_feature=256 four_losses=0.004"

NUM_GPUS=$NUM_GPUS \
NUM_TASK_TOKENS=$NUM_TASK_TOKENS \
BATCH_SIZE=$BATCH_SIZE \
GRAD_ACCUM=$GRAD_ACCUM \
MAX_STEPS=$MAX_STEPS \
SAVE_STEPS=$SAVE_STEPS \
LOG_STEPS=10 \
FULL_FINETUNE=1 \
NUM_WORKERS=4 \
LR=3e-5 \
HEAD_LR=3e-4 \
WARMUP_STEPS=1000 \
PY=$PY \
WEIGHTS=$WEIGHTS \
MODEL_PATH=$WEIGHTS/Qwen3-VL-4B-lingbot-vlm \
LINGBOT_6B=$WEIGHTS/lingbot-vla-v2-6b \
HEAD_WARMSTART_CKPT=$WEIGHTS/lingbot_align_heads_warmstart \
DEPTH_MOGE_PATH=$WEIGHTS/moge-2-vitb-normal/model.pt \
DEPTH_MORGBD_PATH=$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt \
LINGBOT_SRC_ROOT=$ROOT/code/lingbot-vla-v2 \
UTILS3D_MOGE_PATH=$ROOT/py_deps/utils3d_moge \
FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
FASTWAM_DATASET_DIRS=$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot \
FASTWAM_TEXT_EMBEDDING_CACHE_DIR=$ROOT/data/libero_qwen \
FASTWAM_PRETRAINED_NORM_STATS=$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json \
OUTPUT_DIR=$OUTPUT_DIR \
bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1.sh
