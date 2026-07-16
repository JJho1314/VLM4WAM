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
GPU_POOL="${GPU_POOL:-0,1,2,3,4,5,7}"
TARGET_PER_GPU="${TARGET_PER_GPU:-8}"
MAX_PAIRS_PER_PROC="${MAX_PAIRS_PER_PROC:-8}"
MAX_LAUNCH_PER_TICK="${MAX_LAUNCH_PER_TICK:-64}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-90}"
STEP="${STEP:-14800}"
NIS="${NIS:-10}"
TAG_BASE="${TAG_BASE:-plus_autofill_$(date +%Y%m%d_%H%M%S)}"
EXCLUDE_CATEGORIES="${EXCLUDE_CATEGORIES:-}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

if [ -z "$OUT" ]; then
  echo "[plus-autofill] ERROR: OUT is empty" >&2
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

mkdir -p "$OUT"

tick=0
while true; do
  tick=$((tick + 1))
  tag="${TAG_BASE}_t${tick}_$(date +%Y%m%d_%H%M%S)"
  shard_dir="$OUT/shards_${tag}"
  meta_file="$OUT/autofill_${tag}.meta"
  rm -rf "$shard_dir"
  mkdir -p "$shard_dir"

  "$PY" - "$LIBERO_PLUS_ROOT" "$OUT" "$SUITES" "$GPU_POOL" "$TARGET_PER_GPU" \
    "$MAX_PAIRS_PER_PROC" "$MAX_LAUNCH_PER_TICK" "$shard_dir" "$meta_file" <<'PYEOF'
import glob
import json
import os
import sys
from collections import Counter, defaultdict

root, out, suites_arg, gpu_pool_arg, target_arg, max_pairs_arg, max_launch_arg, shard_dir, meta_file = sys.argv[1:]
suites = [s.strip() for s in suites_arg.split(",") if s.strip()]
gpu_pool = [g.strip() for g in gpu_pool_arg.split(",") if g.strip()]
target = int(target_arg)
max_pairs = int(max_pairs_arg)
max_launch = int(max_launch_arg)

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

active_pairs = set()
active_by_gpu = Counter()
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        parts = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "ignore").split("\0")
        env = open(f"/proc/{pid}/environ", "rb").read().decode("utf-8", "ignore").split("\0")
    except Exception:
        continue
    cmd = " ".join(parts)
    if "experiments/libero/cosmos_eval_libero_plus.py" not in cmd or out not in cmd:
        continue
    gpu = "?"
    for item in env:
        if item.startswith("CUDA_VISIBLE_DEVICES="):
            gpu = item.split("=", 1)[1]
            break
    active_by_gpu[gpu] += 1
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
                    active_pairs.add(pair)

missing_by_suite = defaultdict(list)
for pair in expected:
    if pair not in seen and pair not in active_pairs:
        missing_by_suite[pair[0]].append(pair)

mixed = []
max_len = max((len(missing_by_suite[s]) for s in suites), default=0)
for idx in range(max_len):
    for suite in suites:
        bucket = missing_by_suite[suite]
        if idx < len(bucket):
            mixed.append(bucket[idx])

slots = []
for gpu in gpu_pool:
    slots.extend([gpu] * max(0, target - active_by_gpu.get(gpu, 0)))
slots = slots[:max_launch]

launch_count = min(len(slots), (len(mixed) + max_pairs - 1) // max_pairs if mixed else 0)
for idx in range(launch_count):
    chunk = mixed[idx * max_pairs:(idx + 1) * max_pairs]
    with open(os.path.join(shard_dir, f"shard_{idx}.txt"), "w", encoding="utf-8") as f:
        f.write(",".join(f"{suite}:{task_id}" for suite, task_id in chunk))

meta = {
    "expected": len(expected),
    "seen": len(seen),
    "missing_not_active": len(mixed),
    "active_pairs": len(active_pairs),
    "active_by_gpu": dict(sorted(active_by_gpu.items())),
    "slots": slots,
    "launch_count": launch_count,
}
json.dump(meta, open(meta_file, "w", encoding="utf-8"), indent=2)
print(json.dumps(meta, sort_keys=True))
PYEOF

  echo "[plus-autofill] $(date -u '+%Y-%m-%dT%H:%M:%SZ') tag=$tag meta=$(cat "$meta_file")"
  launch_count="$("$PY" - "$meta_file" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1]))["launch_count"])
PYEOF
)"
  seen="$("$PY" - "$meta_file" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["seen"])
PYEOF
)"
  expected="$("$PY" - "$meta_file" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["expected"])
PYEOF
)"
  if [ "$seen" -ge "$expected" ]; then
    echo "[plus-autofill] complete for suites=$SUITES"
    exit 0
  fi

  if [ "$launch_count" -le 0 ]; then
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  for idx in $(seq 0 $((launch_count - 1))); do
    pairs="$(cat "$shard_dir/shard_${idx}.txt" 2>/dev/null || true)"
    [ -z "$pairs" ] && continue
    gpu="$("$PY" - "$meta_file" "$idx" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["slots"][int(sys.argv[2])])
PYEOF
)"
    proc_tag="${tag}_${idx}"
    log="$OUT/proc_${proc_tag}_gpu${gpu}.log"
    echo "[plus-autofill] launch idx=$idx gpu=$gpu pairs=$(awk -F, '{print NF}' <<< "$pairs") log=$log"
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
    pid="$!"
    disown "$pid" 2>/dev/null || true
    sleep "${LAUNCH_STAGGER_SECONDS:-1}"
  done

  sleep "$INTERVAL_SECONDS"
done
