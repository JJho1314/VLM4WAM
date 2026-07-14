#!/usr/bin/env bash
# Build a 10Hz copy of the DROID video dataset used by semantic-plan Cosmos.

#SBATCH --job-name=droid-10hz
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-droid-10hz-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-droid-10hz-%j.err

set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT"
mkdir -p logs

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
CONDA_ENV=${CONDA_ENV:-starVLA}
conda activate "$CONDA_ENV" 2>/dev/null || true
if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/ffmpeg" ]]; then
  export PATH="$CONDA_PREFIX/bin:$PATH"
fi
PY=${PY:-/data/user/jhe724/.conda/envs/starVLA/bin/python}

DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3}
TARGET_FPS=${TARGET_FPS:-10}
TARGET_HEIGHT=${TARGET_HEIGHT:-320}
TARGET_WIDTH=${TARGET_WIDTH:-576}
if [[ "$TARGET_HEIGHT" -gt 0 && "$TARGET_WIDTH" -gt 0 ]]; then
  DEFAULT_OUTPUT_ROOT="${DATASET_ROOT}_${TARGET_FPS}hz_${TARGET_HEIGHT}x${TARGET_WIDTH}"
else
  DEFAULT_OUTPUT_ROOT="${DATASET_ROOT}_${TARGET_FPS}hz"
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}
NUM_WORKERS=${NUM_WORKERS:-12}
CRF=${CRF:-18}
PRESET=${PRESET:-}
OVERWRITE=${OVERWRITE:-0}
SOURCE_FPS=${SOURCE_FPS:-15}
DRY_RUN=${DRY_RUN:-0}
MAX_VIDEOS=${MAX_VIDEOS:-0}
NO_COPY_SIDECARS=${NO_COPY_SIDECARS:-0}

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "ERROR: dataset root not found: $DATASET_ROOT" >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found in PATH" >&2
  exit 2
fi
if ! command -v ffprobe >/dev/null 2>&1; then
  echo "ERROR: ffprobe not found in PATH" >&2
  exit 2
fi

echo "dataset=$DATASET_ROOT"
echo "output=$OUTPUT_ROOT"
echo "target_fps=$TARGET_FPS size=${TARGET_HEIGHT}x${TARGET_WIDTH} workers=$NUM_WORKERS crf=$CRF preset=$PRESET overwrite=$OVERWRITE source_fps=$SOURCE_FPS dry_run=$DRY_RUN max_videos=$MAX_VIDEOS no_copy_sidecars=$NO_COPY_SIDECARS"
echo "ffmpeg=$(command -v ffmpeg)"
echo "ffprobe=$(command -v ffprobe)"

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --output-root "$OUTPUT_ROOT"
  --target-fps "$TARGET_FPS"
  --target-height "$TARGET_HEIGHT"
  --target-width "$TARGET_WIDTH"
  --source-fps "$SOURCE_FPS"
  --num-workers "$NUM_WORKERS"
  --crf "$CRF"
  --preset "$PRESET"
  --max-videos "$MAX_VIDEOS"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "$NO_COPY_SIDECARS" == "1" ]]; then
  ARGS+=(--no-copy-sidecars)
fi

exec "$PY" qwen3_vl_semantic_planner/downsample_video_dataset_to_fps.py "${ARGS[@]}"
