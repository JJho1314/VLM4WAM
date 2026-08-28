#!/usr/bin/env bash
#SBATCH --job-name=cosmos-file547-f0-bottle
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail

VLM4WAM_ROOT=/data/user/jhe724/workspace/VLM4WAM
COSMOS_ROOT="$VLM4WAM_ROOT/third_party/cosmos-predict2.5"
COSMOS_VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
WEIGHTS_DIR=/data/user/jhe724/workspace/weights
SOURCE_VIDEO=/data/user/jhe724/workspace/data/droid_success/videos/observation.images.left_external/chunk-000/file_547.mp4
STAGE="$VLM4WAM_ROOT/eval_inputs_oracle_iter3000_file547_firstframe_bottle_microwave_REPRO"
OUT="$VLM4WAM_ROOT/eval_results_oracle_iter3000_file547_firstframe_bottle_microwave_REPRO"
PLAN="$STAGE/plans/yc74616_s0_oracle.pt"
ENCODER="$VLM4WAM_ROOT/third_party/siglip2-so400m-patch14-384"
PY="$COSMOS_VENV/bin/python"

mkdir -p "$STAGE/plans" "$OUT"

for required in \
  "$SOURCE_VIDEO" \
  "$STAGE/yc74616_s0.json" \
  "$STAGE/first_frames/yc74616_f0.png" \
  "$ENCODER/config.json" \
  "$VLM4WAM_ROOT/outputs/cosmos_semantic_plan/converted_iter3000/model_ema_bf16.pt" \
  "$PY"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: missing required artifact: $required" >&2
    exit 2
  fi
done

module load gcc/11.5 2>/dev/null || true
export VIRTUAL_ENV="$COSMOS_VENV"
export PATH="$COSMOS_VENV/bin:$PATH"
export PYTHONPATH="$COSMOS_ROOT:${PYTHONPATH:-}"
export COSMOS_HF_LOCAL_DIRS="$WEIGHTS_DIR"
export COSMOS_LOCAL_MODEL_DIR="$WEIGHTS_DIR/Cosmos-Predict2.5-2B"

NV="$COSMOS_VENV/lib/python3.10/site-packages/nvidia"
if [[ -d "$NV" ]]; then
  export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/cublas/lib:$NV/cusparse/lib:$NV/cusolver/lib:$NV/cufft/lib:$NV/curand/lib:$NV/nccl/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi

cd "$COSMOS_ROOT"
echo "=== export matching-task oracle plan: file_547 frame 4736 (315.733333s), 49 frames ==="
"$PY" scripts/export_semantic_plan_from_video.py \
  --video "$SOURCE_VIDEO" \
  --start 4736 \
  --num-frames 49 \
  --frame-stride 1 \
  --num-keyframes 5 \
  --grid-size 0 \
  --encoder-path "$ENCODER" \
  --output "$PLAN"

echo "=== generate from absolute first frame with iter3000 SG-WAM ==="
ORIG="$STAGE" \
OUT="$OUT" \
COSMOS_ROOT="$COSMOS_ROOT" \
COSMOS_VENV="$COSMOS_VENV" \
WEIGHTS_DIR="$WEIGHTS_DIR" \
COSMOS_NUM_FRAMES=49 \
SEMANTIC_PLAN_NUM_KEYFRAMES=5 \
SEMANTIC_PLAN_SPATIAL_GRID=0 \
bash "$VLM4WAM_ROOT/semantic_localization/oracle_repro/reproduce_oracle_yc.sh"
