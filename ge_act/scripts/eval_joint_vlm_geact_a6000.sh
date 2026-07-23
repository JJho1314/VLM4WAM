#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${PY:-/data/LFT-W02_data/.conda/envs/ge-act/bin/python}
LIBERO_ROOT=${LIBERO_ROOT:-/data/LFT-W02_data/junjie/VLA_RL/docker_libero/LIBERO}
CHECKPOINT=${CHECKPOINT:-/data/LFT-W02_data/junjie/weights/joint_vlm_geact_action_k4_50k/step_40000}
CONFIG=${CONFIG:-$ROOT/ge_act/configs/ltx_model/libero/action_model_libero_joint_step40000_eval.yaml}
OUTPUT=${OUTPUT:-/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000}
MODE=${1:-smoke}

if [[ ! -x "$PY" ]]; then
  echo "evaluation Python is not executable: $PY" >&2
  exit 1
fi
if [[ ! -d "$LIBERO_ROOT/libero" ]]; then
  echo "LIBERO checkout is missing: $LIBERO_ROOT" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT/joint_meta.json" ]]; then
  echo "joint checkpoint is incomplete: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "evaluation config is missing: $CONFIG" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT/ge_act:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT"
cd "$ROOT/ge_act"

COMMON=(
  "$PY" "$ROOT/ge_act/experiments/eval_libero_joint.py"
  --config_file "$CONFIG"
  --joint_ckpt_dir "$CHECKPOINT"
  --output_dir "$OUTPUT"
  --device 0
  --exec_step 8
  --threshold 20
)

if [[ "$MODE" == smoke ]]; then
  SMOKE_MAX_TASKS=1 "${COMMON[@]}" \
    --task_suite_name libero_goal \
    --num_trails_per_task 1
elif [[ "$MODE" == full ]]; then
  SMOKE_MAX_TASKS=1 "${COMMON[@]}" \
    --task_suite_name libero_goal \
    --num_trails_per_task 1
  for suite in libero_spatial libero_object libero_goal libero_10; do
    "${COMMON[@]}" \
      --task_suite_name "$suite" \
      --num_trails_per_task 50
  done
else
  echo "usage: $0 smoke|full" >&2
  exit 2
fi
