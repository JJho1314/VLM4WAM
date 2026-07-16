#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"
GPU_POOL="${GPU_POOL:-0,1,2,3,4,5,6,7}"
TARGET_PLUS_PER_GPU="${TARGET_PLUS_PER_GPU:-5}"
MIN_FREE_MB="${MIN_FREE_MB:-10000}"
MIN_FREE_MB_PER_SLOT="${MIN_FREE_MB_PER_SLOT:-9000}"
MAX_NPROC_PER_ROUND="${MAX_NPROC_PER_ROUND:-16}"
MAX_PAIRS_PER_PROC="${MAX_PAIRS_PER_PROC:-4}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
ROUNDS="${ROUNDS:-0}"
INCLUDE_SUITES="${INCLUDE_SUITES:-libero_object}"
REVERSE_MISSING="${REVERSE_MISSING:-0}"
TAG_ROOT="${TAG_ROOT:-plus_adaptive_obj}"
LOG_DIR="${LOG_DIR:-/data/users/junjie/FastWAM_cosmos/evaluate_results/auto_eval_logs}"
OUT="${OUT:-$(cat /data/users/junjie/current_eval_plus_out.txt 2>/dev/null || true)}"

cd "$REPO" || exit 2
mkdir -p "$LOG_DIR"

round=0
while true; do
  round=$((round + 1))
  if [ "$ROUNDS" -gt 0 ] && [ "$round" -gt "$ROUNDS" ]; then
    echo "[plus-adaptive] reached ROUNDS=$ROUNDS"
    exit 0
  fi

  GPU_LIST="$("$PY" - "$GPU_POOL" "$TARGET_PLUS_PER_GPU" "$MIN_FREE_MB" "$MIN_FREE_MB_PER_SLOT" "$MAX_NPROC_PER_ROUND" <<'PYEOF'
import csv
import io
import os
import subprocess
import sys
from collections import Counter

pool = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
target = int(sys.argv[2])
min_free = int(sys.argv[3])
per_slot = int(sys.argv[4])
max_nproc = int(sys.argv[5])

counts = Counter()
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        env = open(f"/proc/{pid}/environ", "rb").read().decode("utf-8", "ignore")
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode("utf-8", "ignore").replace("\0", " ")
    except Exception:
        continue
    if "experiments/libero/cosmos_eval_libero_plus.py " not in cmd or "python3 - <<" in cmd:
        continue
    cuda = "unset"
    for item in env.split("\0"):
        if item.startswith("CUDA_VISIBLE_DEVICES="):
            cuda = item.split("=", 1)[1]
            break
    counts[cuda] += 1

free_by_gpu = {}
try:
    q = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    for row in csv.reader(io.StringIO(q)):
        if len(row) < 3:
            continue
        idx, used, total = [x.strip() for x in row[:3]]
        free_by_gpu[idx] = max(0, int(total) - int(used))
except Exception:
    pass

gpus = []
for gpu in pool:
    count_slots = max(0, target - counts[gpu])
    free = free_by_gpu.get(gpu, 0)
    mem_slots = max(0, (free - min_free) // per_slot) if free else count_slots
    slots = min(count_slots, mem_slots)
    for _ in range(slots):
        gpus.append(gpu)

print(",".join(gpus[:max_nproc]))
PYEOF
)"

  ts="$(date +%Y%m%d_%H%M%S)"
  echo "[plus-adaptive] $(date -u +%Y-%m-%dT%H:%M:%SZ) round=$round gpu_list=${GPU_LIST:-none}"
  if [ -z "$GPU_LIST" ]; then
    sleep "$SLEEP_SECONDS"
    continue
  fi

  nproc="$(awk -F, '{print NF}' <<< "$GPU_LIST")"
  tag="${TAG_ROOT}_r${round}_${ts}"
  round_log="$LOG_DIR/${tag}.log"
  (
    export GPU_LIST="$GPU_LIST"
    export NPROC="$nproc"
    export MAX_PAIRS_PER_PROC="$MAX_PAIRS_PER_PROC"
    export INCLUDE_SUITES="$INCLUDE_SUITES"
    export REVERSE_MISSING="$REVERSE_MISSING"
    export TAG_PREFIX="$tag"
    export OUT="$OUT"
    export PLUS_SET_LIBERO_CONFIG_PATH="${PLUS_SET_LIBERO_CONFIG_PATH:-0}"
    export SET_MUJOCO_EGL_DEVICE_ID="${SET_MUJOCO_EGL_DEVICE_ID:-0}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
    bash scripts/supplement_plus_missing_active_aware.sh
  ) 2>&1 | tee "$round_log"

  if grep -q "nothing missing" "$round_log"; then
    echo "[plus-adaptive] no missing pairs for INCLUDE_SUITES=$INCLUDE_SUITES"
    exit 0
  fi
  sleep "$SLEEP_SECONDS"
done
