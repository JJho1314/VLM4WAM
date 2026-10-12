#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/user/jhe724/workspace/VLM4WAM_robofollow_eval
RUNTIME=/data/user/jhe724/workspace/robofollow_eval_runtime
export ROBOFOLLOW_ROOT="$RUNTIME/RoboTwin"
PIXI=/data/user/jhe724/.pixi/bin/pixi
MANIFEST="$ROOT/autoresearch/runtime/pixi.toml"
TRAIN=/data/user/jhe724/outputs/robofollow_q35/official3750_text_53k_ac32_bs128/2026_10_10_21_42_18
OUT=/data/user/jhe724/outputs/robofollow_q35/eval/official3750_text_53k_ac32_bs128
CONFIG="$ROOT/ge_act/configs/ltx_model/robofollow/q35_text_p256_official3750_53k_hpc3.yaml"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/data/user/jhe724/envs/gcc11/bin:$PATH"
export CC=/data/user/jhe724/envs/gcc11/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/user/jhe724/envs/gcc11/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX="$CXX"
export MAX_JOBS=4 OMP_NUM_THREADS=4 LC_ALL=C.UTF-8
export SETUPTOOLS_SCM_PRETEND_VERSION=0.7.0
export LD_LIBRARY_PATH="/data/user/jhe724/.conda/envs/qwen35/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
mkdir -p "$OUT"
# Driver libraries live on compute nodes; keep the ICD descriptor private.
export VK_ICD_FILENAMES="$OUT/nvidia-${SLURM_JOB_ID}.json"
printf '%s
' '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}' > "$VK_ICD_FILENAMES"
nvidia-smi
git rev-parse HEAD
case "${1:?smoke or full}" in
 smoke)
  "$PIXI" run --manifest-path "$MANIFEST" build-sim
  "$PIXI" run --manifest-path "$MANIFEST" sim scripts/update_embodiment_config_path.py
  "$PIXI" run --manifest-path "$MANIFEST" sim -m robofollow.assets status
  "$PIXI" run --manifest-path "$MANIFEST" sim "$ROOT/scripts/rf_render_check.py"
  for scene in scene1 scene2 scene3 scene4; do
   "$PIXI" run --manifest-path "$MANIFEST" research scripts/rf_bench_eval.py \
    --config "$CONFIG" --checkpoint "$TRAIN/step_35000/diffusion_pytorch_model.safetensors" \
    --out "$OUT/smoke-${SLURM_JOB_ID}/$scene" --rounds 1 --scenes "$scene" --limit 1 \
    --gpus 1 --slots-per-gpu 1 --modes correct
  done
  ;;
 full)
  checkpoint="$TRAIN/step_53000/diffusion_pytorch_model.safetensors"
  test -s "$checkpoint"
  sha256sum "$checkpoint" > "$OUT/checkpoint-${SLURM_JOB_ID}.sha256"
  "$PIXI" run --manifest-path "$MANIFEST" research scripts/rf_bench_eval.py \
   --config "$CONFIG" --checkpoint "$checkpoint" --out "${EVAL_RESUME_OUT:-$OUT/full-${SLURM_JOB_ID}}" \
   --rounds 10 --gpus 8 --slots-per-gpu 1 --modes correct,shuffle
  ;;
 *) exit 2 ;;
esac
