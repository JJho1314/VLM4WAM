#!/usr/bin/env bash
# Compare identical short evaluations on the explicitly selected idle GPU.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${1:?config YAML}
CHECKPOINT=${2:?checkpoint}
OUT=${3:?new output directory}
: "${CUDA_VISIBLE_DEVICES:?select one idle physical GPU}"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]
used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
(( used < 512 )) || { echo 'Selected GPU is occupied; refusing to launch.' >&2; exit 1; }
mkdir "$OUT"
export ROBOFOLLOW_PIXI=/home/LFT-W02/.pixi/bin/pixi
export ROBOFOLLOW_MANIFEST="$ROOT/autoresearch/runtime/a6000/pixi.toml"
export ROBOFOLLOW_OVERLAY=/data/LFT-W02_data/junfeng/robofollow_eval/runtime/overlay_peft_tf5
export ROBOFOLLOW_SIM_MODULE=rf_a6000_sim
export CUDA_HOME=/usr/local/cuda-12.1
export OMP_NUM_THREADS=4 ROBOFOLLOW_PROFILE=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="/data/LFT-W02_data/junfeng/robofollow_eval/runtime/qwen35/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="$OUT/nvidia.json"
printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}' > "$VK_ICD_FILENAMES"
cd "$ROOT"
git rev-parse HEAD > "$OUT/commit.txt"
nvidia-smi > "$OUT/gpu.txt"
"$ROBOFOLLOW_PIXI" run --manifest-path "$ROBOFOLLOW_MANIFEST" sim "$ROOT/scripts/rf_render_check.py"
"$ROBOFOLLOW_PIXI" run --manifest-path "$ROBOFOLLOW_MANIFEST" research scripts/rf_profile_policy.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --out "$OUT/cache-parity.json"
for slots in 1 2; do
  SECONDS=0
  "$ROBOFOLLOW_PIXI" run --manifest-path "$ROBOFOLLOW_MANIFEST" research \
    scripts/rf_bench_eval.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --out "$OUT/slots-$slots" \
    --scenes scene1 --levels L0 --limit 2 --rounds 1 --gpus 1 --slots-per-gpu "$slots" --modes correct
  printf '%s\n' "$SECONDS" > "$OUT/slots-$slots.seconds"
done
