#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
OUT="${OUT:-$(cat /data/users/junjie/current_eval_plus_out.txt 2>/dev/null || true)}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data/users/junjie/LIBERO-plus}"
LIBERO_PLUS_CONFIG="${LIBERO_PLUS_CONFIG:-/data/users/junjie/.libero_plus}"
SUITES="${SUITES:-libero_spatial,libero_object}"
STEP="${STEP:-14800}"
NIS="${NIS:-10}"
NPROC="${NPROC:-28}"
GPU_LIST="${GPU_LIST:-0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,7,7,7,7}"
MAX_PAIRS_PER_PROC="${MAX_PAIRS_PER_PROC:-8}"
MAX_ROUNDS="${MAX_ROUNDS:-80}"
TAG_BASE="${TAG_BASE:-plus_mixed_resume_$(date +%Y%m%d_%H%M%S)}"
LOOP_SLEEP_SECONDS="${LOOP_SLEEP_SECONDS:-5}"
EXCLUDE_CATEGORIES="${EXCLUDE_CATEGORIES:-}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

if [ -z "$OUT" ]; then
  echo "[plus-mixed-loop] ERROR: OUT is empty" >&2
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
export LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -le 0 ]; then
  echo "[plus-mixed-loop] ERROR: empty GPU_LIST=$GPU_LIST" >&2
  exit 2
fi

mkdir -p "$OUT"

for round in $(seq 1 "$MAX_ROUNDS"); do
  tag="${TAG_BASE}_r${round}_$(date +%Y%m%d_%H%M%S)"
  shard_dir="$OUT/shards_${tag}"
  missing_file="$OUT/missing_${tag}.txt"
  rm -rf "$shard_dir"
  mkdir -p "$shard_dir"

  read -r seen expected missing active launched < <(
    "$PY" - "$LIBERO_PLUS_ROOT" "$OUT" "$SUITES" "$NPROC" "$MAX_PAIRS_PER_PROC" "$shard_dir" "$missing_file" <<'PYEOF'
import glob
import json
import os
import sys
from collections import defaultdict

root, out, suites_arg, nproc, max_pairs, shard_dir, missing_file = sys.argv[1:]
suites = [s.strip() for s in suites_arg.split(",") if s.strip()]
nproc = int(nproc)
max_pairs = int(max_pairs)

cls = json.load(open(os.path.join(root, "libero/libero/benchmark/task_classification.json"), encoding="utf-8"))
expected = []
for suite in suites:
    for task_id, _ in enumerate(cls[suite]):
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
    if not isinstance(data, dict):
        continue
    for row in data.get("by_task", []) or []:
        try:
            pair = (row.get("suite"), int(row.get("task_id")))
        except Exception:
            continue
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
    cmd = " ".join(parts)
    if "experiments/libero/cosmos_eval_libero_plus.py" not in cmd or out not in cmd:
        continue
    for idx, part in enumerate(parts):
        if part == "--pairs" and idx + 1 < len(parts):
            for item in parts[idx + 1].split(","):
                if ":" not in item:
                    continue
                suite, task_s = item.split(":", 1)
                try:
                    pair = (suite, int(task_s))
                except ValueError:
                    continue
                if pair in expected_set and pair not in seen:
                    active.add(pair)

missing_by_suite = defaultdict(list)
for pair in expected:
    if pair not in seen and pair not in active:
        missing_by_suite[pair[0]].append(pair)

mixed = []
max_len = max((len(missing_by_suite[s]) for s in suites), default=0)
for idx in range(max_len):
    for suite in suites:
        bucket = missing_by_suite[suite]
        if idx < len(bucket):
            mixed.append(bucket[idx])

with open(missing_file, "w", encoding="utf-8") as f:
    for suite, task_id in mixed:
        f.write(f"{suite}:{task_id}\n")

nproc = max(1, min(nproc, len(mixed))) if mixed else 0
shards = [[] for _ in range(nproc)]
for idx, pair in enumerate(mixed):
    shards[idx % nproc].append(f"{pair[0]}:{pair[1]}")
if max_pairs > 0:
    shards = [shard[:max_pairs] for shard in shards]
for idx, shard in enumerate(shards):
    with open(os.path.join(shard_dir, f"shard_{idx}.txt"), "w", encoding="utf-8") as f:
        f.write(",".join(shard))
print(len(seen), len(expected), sum(len(v) for v in missing_by_suite.values()), len(active), sum(len(s) for s in shards))
PYEOF
  )

  echo "[plus-mixed-loop] $(date -u '+%Y-%m-%dT%H:%M:%SZ') round=$round suites=$SUITES seen=$seen/$expected missing_not_active=$missing active=$active launched=$launched tag=$tag"
  if [ "$missing" -le 0 ]; then
    echo "[plus-mixed-loop] complete for suites=$SUITES"
    exit 0
  fi
  if [ "$launched" -le 0 ]; then
    sleep "$LOOP_SLEEP_SECONDS"
    continue
  fi

  shard_count="$(find "$shard_dir" -maxdepth 1 -name 'shard_*.txt' | wc -l | tr -d ' ')"
  pids=()
  failed=0
  for idx in $(seq 0 $((shard_count - 1))); do
    pairs="$(cat "$shard_dir/shard_${idx}.txt" 2>/dev/null || true)"
    [ -z "$pairs" ] && continue
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    proc_tag="${tag}_${idx}"
    log="$OUT/proc_${proc_tag}_gpu${gpu}.log"
    echo "[plus-mixed-loop] launch idx=$idx gpu=$gpu pairs=$(awk -F, '{print NF}' <<< "$pairs") log=$log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      if [ "${SET_MUJOCO_EGL_DEVICE_ID:-1}" = "1" ]; then
        export MUJOCO_EGL_DEVICE_ID="$gpu"
      else
        unset MUJOCO_EGL_DEVICE_ID
      fi
      "$PY" experiments/libero/cosmos_eval_libero_plus.py \
        --pairs "$pairs" --tag "$proc_tag" --num_trials 1 --num_inference_steps "$NIS" \
        --coupling mot --run_dir "$RUN_DIR" --step "$STEP" --out_dir "$OUT" \
        --exclude_categories "$EXCLUDE_CATEGORIES" \
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
  echo "[plus-mixed-loop] done round=$round failed_procs=$failed"
  sleep "$LOOP_SLEEP_SECONDS"
done

echo "[plus-mixed-loop] max rounds reached" >&2
exit 1
