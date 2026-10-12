#!/usr/bin/env bash
# Paired warm-start experiment; run in an isolated eight-GPU Slurm allocation.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PIXI=/data/user/jhe724/.pixi/bin/pixi
MANIFEST="$ROOT/autoresearch/runtime/pixi.toml"
OUT=/data/user/jhe724/outputs/robofollow_q35/optimization_20261012
CONFIG="$ROOT/ge_act/configs/ltx_model/robofollow"
export ROBOFOLLOW_ROOT=/data/user/jhe724/workspace/robofollow_eval_runtime/RoboTwin
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="/data/user/jhe724/.conda/envs/qwen35/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE LC_ALL=C.UTF-8
[[ ${SLURM_NNODES:?} == 1 ]]
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9
# Never overwrite a prior pilot or silently repeat a completed experiment.
for target in joint/smoke joint/pilot text/pilot eval; do
 test ! -e "$OUT/$target"
done
cd "$ROOT"
git diff --exit-code HEAD
git rev-parse HEAD | tee "$OUT/commit"
nvidia-smi
export VK_ICD_FILENAMES="$OUT/nvidia-${SLURM_JOB_ID}.json"
printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}' > "$VK_ICD_FILENAMES"
"$PIXI" run --manifest-path "$MANIFEST" optimize-train --config_file "$CONFIG/q35_joint_warm53k_smoke_hpc3.yaml"
"$PIXI" run --manifest-path "$MANIFEST" research - <<'PY'
import json
from pathlib import Path
p=Path('/data/user/jhe724/outputs/robofollow_q35/optimization_20261012/joint/smoke')
r=json.loads((p/'joint_gradient_report.json').read_text())
assert r['qwen_frozen'] is True
assert r['head_gradients'] and all(x['finite'] for x in r['head_gradients'].values())
assert any(x['norm'] > 0 for x in r['head_gradients'].values())
t=json.loads((p/'weight_transfer.json').read_text())
assert all(k.startswith(('semantic_adapter.','transformer_blocks.','research_planner_head.')) for k in t['new'])
assert all(k in t['loaded'] for k in ('action_proj_in.weight','action_proj_out.weight'))
print('Joint smoke passed: action weights preserved and planner receives finite, nonzero gradients.')
PY
for mode in text joint; do
 "$PIXI" run --manifest-path "$MANIFEST" optimize-train --config_file "$CONFIG/q35_${mode}_warm53k_5k_hpc3.yaml"
 test -s "$OUT/$mode/pilot/step_5000/diffusion_pytorch_model.safetensors"
done
# Evaluate a matched baseline slice too; additional optimizer steps are a confound.
for mode in baseline text joint; do
 planner=disabled
 if [[ $mode == baseline ]]; then
  config="$CONFIG/q35_text_p256_official3750_53k_hpc3.yaml"
  checkpoint=/data/user/jhe724/outputs/robofollow_q35/official3750_text_53k_ac32_bs128/2026_10_10_21_42_18/step_53000/diffusion_pytorch_model.safetensors
 else
  config="$CONFIG/q35_${mode}_warm53k_5k_hpc3.yaml"
  checkpoint="$OUT/$mode/pilot/step_5000/diffusion_pytorch_model.safetensors"
  if [[ $mode == joint ]]; then planner=predicted; fi
 fi
 "$PIXI" run --manifest-path "$MANIFEST" research scripts/rf_bench_eval.py \
  --config "$config" --checkpoint "$checkpoint" --planner-mode "$planner" \
  --out "$OUT/eval/$mode" --levels L0,L1,L2 --rounds 3 --base-seed 42 \
  --gpus 8 --slots-per-gpu 1 --modes correct
done
