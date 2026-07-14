#!/usr/bin/env bash
# Full stride-specific 93-frame VAE latent cache for the 10Hz 320x576 DROID copy.

#SBATCH --job-name=vae-range-full
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-15%8
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-vae-range-full-%A_%a.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-vae-range-full-%A_%a.err

set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
COSMOS_ROOT=${COSMOS_ROOT:-$VLM4WAM_ROOT/cosmos-predict2.5}
cd "$COSMOS_ROOT"
mkdir -p "$VLM4WAM_ROOT/logs"

module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
CONDA_ENV=${CONDA_ENV:-dreamzero}
conda activate "$CONDA_ENV" 2>/dev/null || true
PY=${PY:-/data/user/jhe724/.conda/envs/dreamzero/bin/python}

export PYTHONPATH="$COSMOS_ROOT/packages/cosmos-cuda:$COSMOS_ROOT:${PYTHONPATH:-}"

DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_10hz_320x576}
NUM_FRAMES=${NUM_FRAMES:-93}
FRAME_STRIDES=${FRAME_STRIDES:-1,2,3}
CACHE_NUM_FRAMES=${CACHE_NUM_FRAMES:-93}
CACHE_STEP_FRAMES=${CACHE_STEP_FRAMES:-0}
CACHES_PER_RANGE=${CACHES_PER_RANGE:-4}
CACHE_SEED=${CACHE_SEED:-20260701}
WINDOWS_PER_CACHE=${WINDOWS_PER_CACHE:-1}
WINDOW_SEED=${WINDOW_SEED:-20260701}
HEIGHT=${HEIGHT:-320}
WIDTH=${WIDTH:-576}
NUM_SHARDS=${NUM_SHARDS:-16}
TOTAL_CACHE_RECORDS=${TOTAL_CACHE_RECORDS:-}
OUTPUT_DIR=${OUTPUT_DIR:-$DATASET_ROOT/vae_range_latents_wan2pt1_t${NUM_FRAMES}_s${FRAME_STRIDES//,/}_c${CACHE_NUM_FRAMES}_r${CACHES_PER_RANGE}_full}
VAE_PTH=${VAE_PTH:-/data/user/jhe724/workspace/weights/Cosmos-Predict2.5-2B/tokenizer.pth}
OUTPUT_DTYPE=${OUTPUT_DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-0}

export DATASET_ROOT OUTPUT_DIR FRAME_STRIDES NUM_FRAMES CACHE_NUM_FRAMES CACHE_STEP_FRAMES CACHES_PER_RANGE CACHE_SEED

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT/videos" ]]; then
  echo "ERROR: dataset videos not found: $DATASET_ROOT/videos" >&2
  exit 2
fi
if [[ ! -e "$DATASET_ROOT/frame_ranges.json" ]]; then
  echo "ERROR: frame_ranges.json not found: $DATASET_ROOT/frame_ranges.json" >&2
  exit 2
fi
if [[ ! -e "$VAE_PTH" ]]; then
  echo "ERROR: VAE checkpoint not found: $VAE_PTH" >&2
  exit 2
fi

if [[ -z "$TOTAL_CACHE_RECORDS" ]]; then
  TOTAL_CACHE_RECORDS=$("$PY" - <<'PY'
import importlib.util
import os
from pathlib import Path

script = Path("scripts/precompute_cosmos_vae_range_latents.py")
spec = importlib.util.spec_from_file_location("vae_range_count", script)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

dataset_root = Path(os.environ["DATASET_ROOT"])
records = mod.make_cache_records_from_frame_ranges(
    mod.load_frame_ranges(dataset_root / "frame_ranges.json"),
    output_dir=Path(os.environ["OUTPUT_DIR"]),
    frame_strides=mod.parse_int_list(os.environ["FRAME_STRIDES"]),
    min_sequence_length=int(os.environ["NUM_FRAMES"]),
    cache_num_frames=int(os.environ["CACHE_NUM_FRAMES"]),
    cache_step_frames=int(os.environ["CACHE_STEP_FRAMES"]),
    caches_per_range=int(os.environ["CACHES_PER_RANGE"]),
    cache_seed=int(os.environ["CACHE_SEED"]),
)
print(len(records))
PY
)
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SHARD_SIZE=$(( (TOTAL_CACHE_RECORDS + NUM_SHARDS - 1) / NUM_SHARDS ))
START_CACHE_INDEX=$(( TASK_ID * SHARD_SIZE ))
MAX_CACHE_RECORDS=$SHARD_SIZE

echo "mode=Cosmos Wan2.1 range VAE latent full"
echo "task=$TASK_ID/$NUM_SHARDS start_cache_index=$START_CACHE_INDEX max_cache_records=$MAX_CACHE_RECORDS total_cache_records=$TOTAL_CACHE_RECORDS"
echo "dataset=$DATASET_ROOT"
echo "output=$OUTPUT_DIR"
echo "frames=$NUM_FRAMES strides=$FRAME_STRIDES cache_num_frames=$CACHE_NUM_FRAMES caches_per_range=$CACHES_PER_RANGE windows_per_cache=$WINDOWS_PER_CACHE"
echo "size=${HEIGHT}x${WIDTH} vae_pth=$VAE_PTH dtype=$OUTPUT_DTYPE overwrite=$OVERWRITE"

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --frame-ranges "$DATASET_ROOT/frame_ranges.json"
  --frame-strides "$FRAME_STRIDES"
  --num-frames "$NUM_FRAMES"
  --cache-num-frames "$CACHE_NUM_FRAMES"
  --cache-step-frames "$CACHE_STEP_FRAMES"
  --caches-per-range "$CACHES_PER_RANGE"
  --cache-seed "$CACHE_SEED"
  --windows-per-cache "$WINDOWS_PER_CACHE"
  --window-seed "$WINDOW_SEED"
  --height "$HEIGHT"
  --width "$WIDTH"
  --start-cache-index "$START_CACHE_INDEX"
  --max-cache-records "$MAX_CACHE_RECORDS"
  --vae-pth "$VAE_PTH"
  --output-dtype "$OUTPUT_DTYPE"
  --cache-manifest-name "cache_manifest_${TASK_ID}.jsonl"
  --window-manifest-name "window_manifest_${TASK_ID}.jsonl"
  --summary-name "summary_${TASK_ID}.json"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

exec "$PY" scripts/precompute_cosmos_vae_range_latents.py "${ARGS[@]}"
