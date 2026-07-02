#!/usr/bin/env bash

#SBATCH --job-name=cosmos-semplan
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-cosmos-semplan-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-cosmos-semplan-%j.err

set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
COSMOS_ROOT=${COSMOS_ROOT:-$VLM4WAM_ROOT/third_party/cosmos-predict2.5}
cd "$COSMOS_ROOT"
mkdir -p "$VLM4WAM_ROOT/logs"

module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
CONDA_ENV=${CONDA_ENV:-cosmos_fw}
conda activate "$CONDA_ENV" 2>/dev/null || true
COSMOS_VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
if [[ -z "${PY:-}" && -x "$COSMOS_VENV/bin/python" ]]; then
  PY="$COSMOS_VENV/bin/python"
elif [[ -z "${PY:-}" && -x "$COSMOS_ROOT/.venv/bin/python" ]]; then
  PY="$COSMOS_ROOT/.venv/bin/python"
fi
PY=${PY:-/data/user/jhe724/.conda/envs/cosmos_fw/bin/python}
if [[ -n "${COSMOS_VENV:-}" && -d "$COSMOS_VENV" ]]; then
  export VIRTUAL_ENV="$COSMOS_VENV"
  export PATH="$COSMOS_VENV/bin:$PATH"
  NV_LIB="$COSMOS_VENV/lib/python3.10/site-packages/nvidia"
  if [[ -d "$NV_LIB" ]]; then
    export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
  fi
fi
export PYTHONPATH="$COSMOS_ROOT:${PYTHONPATH:-}"
export COSMOS_HF_LOCAL_DIRS=${COSMOS_HF_LOCAL_DIRS:-/data/user/jhe724/workspace/weights}
export COSMOS_LOCAL_MODEL_DIR=${COSMOS_LOCAL_MODEL_DIR:-/data/user/jhe724/workspace/weights/Cosmos-Predict2.5-2B}
export COSMOS_DISABLE_CUDNN_SDPA=${COSMOS_DISABLE_CUDNN_SDPA:-1}
export COSMOS_DISABLE_CUDNN_CONV=${COSMOS_DISABLE_CUDNN_CONV:-1}

export DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3}
export COSMOS_NUM_FRAMES=${COSMOS_NUM_FRAMES:-93}
export VIDEO_HEIGHT=${VIDEO_HEIGHT:-320}
export VIDEO_WIDTH=${VIDEO_WIDTH:-576}
export SEMANTIC_PLAN_DIM=${SEMANTIC_PLAN_DIM:-1152}
export SEMANTIC_PLAN_NUM_KEYFRAMES=${SEMANTIC_PLAN_NUM_KEYFRAMES:-6}
export SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES=${SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES:-16}
export SEMANTIC_PLAN_SPATIAL_GRID=${SEMANTIC_PLAN_SPATIAL_GRID:-9}
export SEMANTIC_PLAN_DROPOUT_PROB=${SEMANTIC_PLAN_DROPOUT_PROB:-0.15}
# SEMANTIC_PLAN_ONLINE=1 encodes SigLIP2 plans on the fly (no .pt features read; manifest
# still provides windows + VAE latents). Combine with SEMANTIC_PLAN_SPATIAL_GRID=0 for the
# native 27x27 grid.
export SEMANTIC_PLAN_ONLINE=${SEMANTIC_PLAN_ONLINE:-0}
export SEMANTIC_PLAN_ONLINE_ENCODER_PATH=${SEMANTIC_PLAN_ONLINE_ENCODER_PATH:-/data/user/jhe724/workspace/VLM4WAM/third_party/siglip2-so400m-patch14-384}
export SEMANTIC_PLAN_ONLINE_NUM_KEYFRAMES=${SEMANTIC_PLAN_ONLINE_NUM_KEYFRAMES:-$SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES}
export SEMANTIC_PLAN_FRAME_STRIDES=${SEMANTIC_PLAN_FRAME_STRIDES:-1,2,3}
export SEMANTIC_PLAN_WINDOW_STRIDE=${SEMANTIC_PLAN_WINDOW_STRIDE:-24}
if [[ "$SEMANTIC_PLAN_SPATIAL_GRID" -le 0 ]]; then
  SEMANTIC_PLAN_GRID_TAG=gnative
  SEMANTIC_PLAN_SOURCE_TOKENS=0
else
  SEMANTIC_PLAN_GRID_TAG=g${SEMANTIC_PLAN_SPATIAL_GRID}
  SEMANTIC_PLAN_SOURCE_TOKENS=$((SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES * SEMANTIC_PLAN_SPATIAL_GRID * SEMANTIC_PLAN_SPATIAL_GRID))
fi
SEMANTIC_PLAN_STRIDE_TAG=${SEMANTIC_PLAN_FRAME_STRIDES//,/}
export SEMANTIC_PLAN_DATASET_MAX_TOKENS=${SEMANTIC_PLAN_DATASET_MAX_TOKENS:-$SEMANTIC_PLAN_SOURCE_TOKENS}
export SEMANTIC_PLAN_DIR=${SEMANTIC_PLAN_DIR:-$DATASET_ROOT/siglip2_semantic_plan_k${SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES}_${SEMANTIC_PLAN_GRID_TAG}_cosmos_t${COSMOS_NUM_FRAMES}_s${SEMANTIC_PLAN_STRIDE_TAG}_step${SEMANTIC_PLAN_WINDOW_STRIDE}_full}
export SEMANTIC_PLAN_MANIFEST=${SEMANTIC_PLAN_MANIFEST:-manifest*.jsonl}
export VAE_LATENT_DIR=${VAE_LATENT_DIR:-$DATASET_ROOT/vae_range_latents_wan2pt1_t${COSMOS_NUM_FRAMES}_s${SEMANTIC_PLAN_STRIDE_TAG}_c${COSMOS_NUM_FRAMES}_r4_full}
export VAE_LATENT_MANIFEST=${VAE_LATENT_MANIFEST:-window_manifest*.jsonl}
export CHECKPOINT_LOAD_PATH=${CHECKPOINT_LOAD_PATH:-/data/user/jhe724/workspace/weights/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$VLM4WAM_ROOT/outputs/cosmos_semantic_plan}
export JOB_NAME=${JOB_NAME:-2b_droid_semantic_plan_320x576_93f_k${SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES}${SEMANTIC_PLAN_GRID_TAG}_win_bs2acc8_gbs128_step3000}
export WANDB_PROJECT=${WANDB_PROJECT:-cosmos_predict_v2p5}
export WANDB_GROUP=${WANDB_GROUP:-semantic_plan_video2world}
export WANDB_MODE=${WANDB_MODE:-online}

export BATCH_SIZE=${BATCH_SIZE:-2}
export GRAD_ACCUM_ITER=${GRAD_ACCUM_ITER:-8}
export NUM_GPUS=${NUM_GPUS:-8}
export MAX_ITER=${MAX_ITER:-3000}
export SAVE_ITER=${SAVE_ITER:-500}
export TRAIN_SAMPLE_EVERY=${TRAIN_SAMPLE_EVERY:-0}
export LOGGING_ITER=${LOGGING_ITER:-20}
export NUM_WORKERS=${NUM_WORKERS:-8}
export PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
export PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
export LR=${LR:-4.315837287515549e-05}
export WARMUP_STEPS=${WARMUP_STEPS:-500}
export MASTER_PORT=${MASTER_PORT:-29433}

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
"$PY" - <<'PY'
missing = []
for name in ("transformer_engine.pytorch", "flash_attn"):
    try:
        __import__(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("ERROR: Cosmos training environment is missing dependencies:\n" + "\n".join(missing))
print("python_env_ok", flush=True)
PY
export SEMANTIC_PLAN_EPISODE_SAMPLING=${SEMANTIC_PLAN_EPISODE_SAMPLING:-0}
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: dataset root not found: $DATASET_ROOT" >&2
  exit 2
fi
if [[ "$SEMANTIC_PLAN_EPISODE_SAMPLING" == "1" ]]; then
  if [[ ! -f "${FRAME_RANGES_JSON:-$DATASET_ROOT/frame_ranges.json}" ]]; then
    echo "ERROR: frame_ranges.json not found for episode sampling" >&2
    exit 2
  fi
elif [[ ! -d "$SEMANTIC_PLAN_DIR" ]]; then
  echo "ERROR: semantic plan dir not found: $SEMANTIC_PLAN_DIR" >&2
  exit 2
fi
if [[ -n "$VAE_LATENT_DIR" && "$VAE_LATENT_DIR" != "none" && ! -d "$VAE_LATENT_DIR" ]]; then
  echo "ERROR: VAE latent dir not found: $VAE_LATENT_DIR" >&2
  exit 2
fi
MANIFEST_CANDIDATES=()
if [[ "$SEMANTIC_PLAN_EPISODE_SAMPLING" != "1" ]]; then
IFS=',' read -ra MANIFEST_ITEMS <<< "$SEMANTIC_PLAN_MANIFEST"
for item in "${MANIFEST_ITEMS[@]}"; do
  item="${item#"${item%%[![:space:]]*}"}"
  item="${item%"${item##*[![:space:]]}"}"
  [[ -z "$item" ]] && continue
  if [[ "$item" = /* ]]; then
    pattern="$item"
  else
    pattern="$SEMANTIC_PLAN_DIR/$item"
  fi
  if [[ "$pattern" == *[\*\?\[]* ]]; then
    while IFS= read -r candidate; do
      MANIFEST_CANDIDATES+=("$candidate")
    done < <(compgen -G "$pattern" | sort)
  elif [[ -f "$pattern" ]]; then
    MANIFEST_CANDIDATES+=("$pattern")
  fi
done
if [[ "${#MANIFEST_CANDIDATES[@]}" -eq 0 ]]; then
  echo "ERROR: semantic plan manifest not found: $SEMANTIC_PLAN_DIR/$SEMANTIC_PLAN_MANIFEST" >&2
  exit 2
fi
fi
if [[ ! -e "$CHECKPOINT_LOAD_PATH" ]]; then
  echo "ERROR: checkpoint load path not found: $CHECKPOINT_LOAD_PATH" >&2
  exit 2
fi

echo "mode=Cosmos Video2World semantic-plan oracle"
echo "cosmos_root=$COSMOS_ROOT"
echo "dataset=$DATASET_ROOT"
echo "semantic_plan_dir=$SEMANTIC_PLAN_DIR"
echo "semantic_plan_manifest=$SEMANTIC_PLAN_MANIFEST count=${#MANIFEST_CANDIDATES[@]}"
echo "vae_latent_dir=$VAE_LATENT_DIR"
echo "vae_latent_manifest=$VAE_LATENT_MANIFEST"
echo "checkpoint=$CHECKPOINT_LOAD_PATH"
echo "output_root=$IMAGINAIRE_OUTPUT_ROOT"
echo "disable_cudnn_sdpa=$COSMOS_DISABLE_CUDNN_SDPA"
echo "frames=$COSMOS_NUM_FRAMES size=${VIDEO_HEIGHT}x${VIDEO_WIDTH}"
echo "semantic_plan=dim${SEMANTIC_PLAN_DIM}_sourcek${SEMANTIC_PLAN_SOURCE_NUM_KEYFRAMES}_targetk${SEMANTIC_PLAN_NUM_KEYFRAMES}_g${SEMANTIC_PLAN_SPATIAL_GRID}_dataset_max_tokens${SEMANTIC_PLAN_DATASET_MAX_TOKENS}"
echo "batch_per_gpu=$BATCH_SIZE accum=$GRAD_ACCUM_ITER num_gpus=$NUM_GPUS global_batch=$((BATCH_SIZE * GRAD_ACCUM_ITER * NUM_GPUS))"
echo "max_iter=$MAX_ITER save_iter=$SAVE_ITER train_sample_every=$TRAIN_SAMPLE_EVERY lr=$LR"

"$PY" - <<'PY'
import json
import os
import glob
from pathlib import Path

import torch

if os.environ.get("SEMANTIC_PLAN_EPISODE_SAMPLING", "0") == "1":
    ranges_path = os.environ.get(
        "FRAME_RANGES_JSON", os.path.join(os.environ["DATASET_ROOT"], "frame_ranges.json")
    )
    ranges = json.loads(Path(ranges_path).read_text())
    print(json.dumps({"episode_sampling": True, "num_stems": len(ranges)}), flush=True)
    raise SystemExit(0)

label_dir = Path(os.environ["SEMANTIC_PLAN_DIR"])
manifest_items = [item.strip() for item in os.environ.get("SEMANTIC_PLAN_MANIFEST", "").split(",") if item.strip()]
manifest_paths = []
for item in manifest_items:
    path = Path(item) if Path(item).is_absolute() else label_dir / item
    pattern = str(path)
    if any(char in pattern for char in "*?[]"):
        manifest_paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    elif path.exists():
        manifest_paths.append(path)
if not manifest_paths:
    raise SystemExit(f"No semantic-plan manifests found in {label_dir}")
with manifest_paths[0].open("r", encoding="utf-8") as f:
    first_record = json.loads(next(line for line in f if line.strip()))
if os.environ.get("SEMANTIC_PLAN_ONLINE", "0") == "1":
    # Online encoding reads no .pt features; the manifest only defines windows.
    print(json.dumps({
        "label_dir": str(label_dir),
        "num_manifest": len(manifest_paths),
        "online_encoding": True,
    }), flush=True)
else:
    sample_path = first_record.get("path")
    if sample_path:
        sample_path = Path(sample_path)
    else:
        sample_path = label_dir / f"{first_record['sample_id']}.pt"
    if not sample_path.exists():
        raise SystemExit(f"Sample semantic-plan file not found: {sample_path}")
    payload = torch.load(sample_path, map_location="cpu", weights_only=False)
    plan = payload["semantic_plan"] if isinstance(payload, dict) and "semantic_plan" in payload else payload
    print(json.dumps({
        "label_dir": str(label_dir),
        "num_manifest": len(manifest_paths),
        "sample": sample_path.name,
        "semantic_plan_shape": list(plan.shape),
    }), flush=True)
PY

exec "$PY" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- \
  experiment=predict2_video2world_training_2b_droid_semantic_plan_320x576_93f
