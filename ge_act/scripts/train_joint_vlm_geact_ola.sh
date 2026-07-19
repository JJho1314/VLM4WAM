#!/usr/bin/env bash
# Formal and bounded-smoke launcher for joint K4 Qwen3-VL + GE-Act on OLA.
set -euo pipefail

GE_ACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname "$GE_ACT_ROOT")"
CALLER_PWD="$PWD"
CONFIG="${CONFIG:-${1:-$GE_ACT_ROOT/configs/ltx_model/libero/video_model_libero_joint_vlm_geact_k4_predecoded.yaml}}"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$CALLER_PWD/$CONFIG"
fi
CONFIG="$(realpath -m -- "$CONFIG")"

PY="${PY:-/data/users/junjie/envs/vlm4wam/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "$PY")/torchrun}"
RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=8

export PYTHONPATH="$REPO_ROOT:$GE_ACT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -x "$PY" ]]; then
  echo "OLA Python is not executable: $PY" >&2
  exit 2
fi
if [[ ! -x "$TORCHRUN" ]]; then
  echo "OLA torchrun is not executable: $TORCHRUN" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "joint training config does not exist: $CONFIG" >&2
  exit 2
fi

MAIN_ARGS=(--config_file "$CONFIG")
if [[ "$RUN_KIND" == "smoke" ]]; then
  NUM_GPUS=1
  MAIN_ARGS+=(
    --max_train_steps 1
    --batch_size_override 1
    --gradient_accumulation_steps_override 1
    --disable_deepspeed
    --enable_8bit_optimizer
  )
elif [[ "$RUN_KIND" == "smoke8" ]]; then
  MAIN_ARGS+=(
    --max_train_steps 10
    --batch_size_override 1
    --gradient_accumulation_steps_override 1
  )
elif [[ "$RUN_KIND" != "formal" ]]; then
  echo "RUN_KIND must be formal, smoke, or smoke8, got '$RUN_KIND'" >&2
  exit 2
fi

cd "$GE_ACT_ROOT"

# This is verification only. require_predecoded=true makes any online decode a
# hard error in the dataset, while this command proves every cached episode exists.
"$PY" scripts/predecode_lerobot_videos.py \
  --config "$CONFIG" \
  --verify-only

# Validate the immutable formal eight-GPU recipe even for a one-GPU smoke; smoke
# changes are CLI-only and never mutate the checked-in YAML.
"$PY" scripts/preflight_ltx_siglip2.py \
  --config "$CONFIG" \
  --world-size 8 \
  --require-joint-formal

exec "$TORCHRUN" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NUM_GPUS" \
  main.py \
  "${MAIN_ARGS[@]}"
