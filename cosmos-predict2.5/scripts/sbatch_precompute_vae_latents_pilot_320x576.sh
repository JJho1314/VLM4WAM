#!/usr/bin/env bash
# Pilot cache for frozen Cosmos Wan2.1 VAE latents on the 10Hz 320x576 DROID copy.

#SBATCH --job-name=vae-latent-pilot
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-vae-latent-pilot-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-vae-latent-pilot-%j.err

set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
COSMOS_ROOT=${COSMOS_ROOT:-$VLM4WAM_ROOT/cosmos-predict2.5}
cd "$COSMOS_ROOT"
mkdir -p "$VLM4WAM_ROOT/logs"

module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
CONDA_ENV=${CONDA_ENV:-starVLA}
conda activate "$CONDA_ENV" 2>/dev/null || true
PY=${PY:-/data/user/jhe724/.conda/envs/starVLA/bin/python}

export PYTHONPATH="$COSMOS_ROOT/packages/cosmos-cuda:$COSMOS_ROOT:${PYTHONPATH:-}"

DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_10hz_320x576}
NUM_FRAMES=${NUM_FRAMES:-93}
FRAME_STRIDE=${FRAME_STRIDE:-1}
HEIGHT=${HEIGHT:-320}
WIDTH=${WIDTH:-576}
MAX_SAMPLES=${MAX_SAMPLES:-100}
START_INDEX=${START_INDEX:-0}
MAX_WINDOWS_PER_RANGE=${MAX_WINDOWS_PER_RANGE:-1}
OUTPUT_DIR=${OUTPUT_DIR:-$DATASET_ROOT/vae_latents_wan2pt1_t${NUM_FRAMES}_s${FRAME_STRIDE}_pilot${MAX_SAMPLES}}
VAE_PTH=${VAE_PTH:-/data/user/jhe724/workspace/weights/Cosmos-Predict2.5-2B/tokenizer.pth}
OUTPUT_DTYPE=${OUTPUT_DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-0}

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

"$PY" - <<'PY'
import importlib
for name in ("torch", "decord", "torchvision"):
    importlib.import_module(name)
print("python_env_ok", flush=True)
PY

echo "mode=Cosmos Wan2.1 VAE latent pilot"
echo "cosmos_root=$COSMOS_ROOT"
echo "dataset=$DATASET_ROOT"
echo "output=$OUTPUT_DIR"
echo "frames=$NUM_FRAMES stride=$FRAME_STRIDE size=${HEIGHT}x${WIDTH}"
echo "max_samples=$MAX_SAMPLES start_index=$START_INDEX max_windows_per_range=$MAX_WINDOWS_PER_RANGE"
echo "vae_pth=$VAE_PTH dtype=$OUTPUT_DTYPE"

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --frame-ranges "$DATASET_ROOT/frame_ranges.json"
  --num-frames "$NUM_FRAMES"
  --frame-stride "$FRAME_STRIDE"
  --max-windows-per-range "$MAX_WINDOWS_PER_RANGE"
  --height "$HEIGHT"
  --width "$WIDTH"
  --start-index "$START_INDEX"
  --max-samples "$MAX_SAMPLES"
  --vae-pth "$VAE_PTH"
  --output-dtype "$OUTPUT_DTYPE"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

exec "$PY" scripts/precompute_cosmos_vae_latents_pilot.py "${ARGS[@]}"
