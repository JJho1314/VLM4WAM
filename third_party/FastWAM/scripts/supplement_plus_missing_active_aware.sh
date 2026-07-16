#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
OUT="${OUT:-$(cat /data/users/junjie/current_eval_plus_out.txt 2>/dev/null || true)}"
STEP="${STEP:-14800}"
NIS="${NIS:-10}"
NPROC="${NPROC:-8}"
GPU_LIST="${GPU_LIST:-0,2}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data/users/junjie/LIBERO-plus}"
LIBERO_PLUS_CONFIG="${LIBERO_PLUS_CONFIG:-/data/users/junjie/.libero_plus}"
PLUS_SET_LIBERO_CONFIG_PATH="${PLUS_SET_LIBERO_CONFIG_PATH:-1}"
EXCL="${EXCL:-}"
INCLUDE_SUITES="${INCLUDE_SUITES:-}"
EXCLUDE_SUITES="${EXCLUDE_SUITES:-}"
TAG_PREFIX="${TAG_PREFIX:-plus_active_supp_$(date +%Y%m%d_%H%M%S)}"
REVERSE_MISSING="${REVERSE_MISSING:-0}"
MAX_PAIRS_PER_PROC="${MAX_PAIRS_PER_PROC:-4}"
DRY_RUN="${DRY_RUN:-0}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

if [ -z "$OUT" ]; then
  echo "[plus-active-supp] ERROR: OUT is empty and /data/users/junjie/current_eval_plus_out.txt is unavailable" >&2
  exit 2
fi

source "$TRAIN_ENV"
cd "$REPO" || exit 2

for d in "$VENV"/lib/python3.10/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export MAGICK_HOME="${MAGICK_HOME:-/data/users/junjie/im_env}"
export LD_LIBRARY_PATH="/data/users/junjie/im_env/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_PLUS_ROOT="$LIBERO_PLUS_ROOT"
if [ "$PLUS_SET_LIBERO_CONFIG_PATH" = "1" ]; then
  export LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG"
else
  unset LIBERO_CONFIG_PATH
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPU="${#GPUS[@]}"
if [ "$NGPU" -le 0 ]; then
  echo "[plus-active-supp] ERROR: empty GPU_LIST=$GPU_LIST" >&2
  exit 2
fi

mkdir -p "$OUT"
MISSING_FILE="$OUT/missing_${TAG_PREFIX}.txt"
SHARD_DIR="$OUT/shards_${TAG_PREFIX}"
rm -rf "$SHARD_DIR"
mkdir -p "$SHARD_DIR"

"$PY" - "$LIBERO_PLUS_ROOT" "$OUT" "$EXCL" "$INCLUDE_SUITES" "$EXCLUDE_SUITES" "$MISSING_FILE" "$NPROC" "$SHARD_DIR" "$REVERSE_MISSING" "$MAX_PAIRS_PER_PROC" <<'PYEOF'
import glob
import json
import os
import sys

root, out, excl, include_suites, exclude_suites, missing_file, nproc, shard_dir, reverse, max_pairs_per_proc = sys.argv[1:]
nproc = int(nproc)
reverse = reverse == "1"
max_pairs_per_proc = int(max_pairs_per_proc)
excluded = {item.strip() for item in excl.split(",") if item.strip()}
included_suites = {item.strip() for item in include_suites.split(",") if item.strip()}
excluded_suites = {item.strip() for item in exclude_suites.split(",") if item.strip()}

cls = json.load(open(os.path.join(root, "libero/libero/benchmark/task_classification.json"), encoding="utf-8"))
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
expected = []
for suite in suites:
    if included_suites and suite not in included_suites:
        continue
    if suite in excluded_suites:
        continue
    for task_id, entry in enumerate(cls[suite]):
        if entry["category"] in excluded:
            continue
        expected.append((suite, task_id))

expected_set = set(expected)
seen = set()
for path in sorted(glob.glob(os.path.join(out, "results_*.json")) + glob.glob(os.path.join(out, "results_partial_*.json"))):
    if os.path.basename(path).startswith("summary_"):
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    for row in data.get("by_task", []):
        pair = (row.get("suite"), int(row.get("task_id", -1)))
        if pair in expected_set and int(row.get("trials", 0) or 0) >= 1:
            seen.add(pair)

active = set()
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        parts = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "ignore").split("\0")
    except Exception:
        continue
    if not parts or "experiments/libero/cosmos_eval_libero_plus.py" not in " ".join(parts):
        continue
    for idx, part in enumerate(parts):
        if part == "--pairs" and idx + 1 < len(parts):
            for item in parts[idx + 1].split(","):
                if ":" not in item:
                    continue
                suite, task = item.split(":", 1)
                try:
                    pair = (suite, int(task))
                except ValueError:
                    continue
                if pair in expected_set:
                    active.add(pair)

missing_pairs = [pair for pair in expected if pair not in seen and pair not in active]
if reverse:
    missing_pairs = list(reversed(missing_pairs))
missing = [f"{suite}:{task_id}" for suite, task_id in missing_pairs]

with open(missing_file, "w", encoding="utf-8") as f:
    for pair in missing:
        f.write(pair + "\n")

nproc = max(1, min(nproc, len(missing))) if missing else 0
shards = [[] for _ in range(nproc)]
for idx, pair in enumerate(missing):
    shards[idx % nproc].append(pair)
if max_pairs_per_proc > 0:
    shards = [shard[:max_pairs_per_proc] for shard in shards]
for idx, shard in enumerate(shards):
    with open(os.path.join(shard_dir, f"shard_{idx}.txt"), "w", encoding="utf-8") as f:
        f.write(",".join(shard))
launched = sum(len(shard) for shard in shards)
print(
    f"[plus-active-supp] expected={len(expected)} seen={len(seen)} active={len(active)} "
    f"missing_after_active={len(missing)} nproc={nproc} reverse={reverse} "
    f"max_pairs_per_proc={max_pairs_per_proc} launched_pairs={launched}"
)
PYEOF

PAIR_COUNT="$(wc -l < "$MISSING_FILE" | tr -d ' ')"
if [ "$PAIR_COUNT" -le 0 ]; then
  echo "[plus-active-supp] nothing missing"
  exit 0
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "[plus-active-supp] dry run only: missing_file=$MISSING_FILE shard_dir=$SHARD_DIR"
  exit 0
fi

SHARD_COUNT="$(find "$SHARD_DIR" -maxdepth 1 -name 'shard_*.txt' | wc -l | tr -d ' ')"
echo "[plus-active-supp] launching $SHARD_COUNT procs on gpus=$GPU_LIST out=$OUT tag=$TAG_PREFIX"

pids=()
failed=0
for idx in $(seq 0 $((SHARD_COUNT - 1))); do
  pairs="$(cat "$SHARD_DIR/shard_${idx}.txt" 2>/dev/null || true)"
  [ -z "$pairs" ] && continue
  gpu="${GPUS[$((idx % NGPU))]}"
  tag="${TAG_PREFIX}_${idx}"
  log="$OUT/proc_${tag}_gpu${gpu}.log"
  echo "[plus-active-supp] launch idx=$idx gpu=$gpu pairs=$(awk -F, '{print NF}' <<< "$pairs") log=$log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    if [ "${SET_MUJOCO_EGL_DEVICE_ID:-0}" = "1" ]; then
      export MUJOCO_EGL_DEVICE_ID="$gpu"
    else
      unset MUJOCO_EGL_DEVICE_ID
    fi
    "$PY" experiments/libero/cosmos_eval_libero_plus.py \
      --pairs "$pairs" --tag "$tag" --num_trials 1 --num_inference_steps "$NIS" \
      --coupling mot --run_dir "$RUN_DIR" --step "$STEP" --out_dir "$OUT" --exclude_categories "$EXCL" \
      --action_hidden_dim "$ACTION_HIDDEN_DIM" --action_ffn_dim "$ACTION_FFN_DIM" \
      --action_attention_head_dim "$ACTION_ATTENTION_HEAD_DIM" --no-save_videos
  ) > "$log" 2>&1 &
  pids+=("$!")
  sleep "${LAUNCH_STAGGER_SECONDS:-1}"
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done
echo "[plus-active-supp] done failed_procs=$failed"
exit 0
