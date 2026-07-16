#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data/users/junjie/LIBERO-plus}"
CURRENT_OUT_FILE="${CURRENT_OUT_FILE:-/data/users/junjie/current_eval_plus_out.txt}"
MAIN_PLUS_SCREEN="${MAIN_PLUS_SCREEN:-plus_chunked_gpu012_from14800_20260702_014713}"

NIS="${NIS:-10}"
NPROC="${NPROC:-40}"
GPU_LIST="${GPU_LIST:-3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7}"
MAX_PAIRS_PER_PROC="${MAX_PAIRS_PER_PROC:-4}"
SLEEP_BETWEEN_ROUNDS="${SLEEP_BETWEEN_ROUNDS:-5}"
IDLE_SLEEP="${IDLE_SLEEP:-30}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-1}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

cd "$REPO" || exit 2

parse_step() {
  local out="$1"
  local base
  base="$(basename "$out")"
  sed -n 's/.*_step0*\([0-9][0-9]*\)_libero_plus.*/\1/p' <<< "$base"
}

plus_missing_stats() {
  local out="$1"
  python3 - "$LIBERO_PLUS_ROOT" "$out" <<'PYEOF'
import glob
import json
import os
import sys

root, out = sys.argv[1:]
cls_path = os.path.join(root, "libero/libero/benchmark/task_classification.json")
cls = json.load(open(cls_path, encoding="utf-8"))
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
expected = [(suite, idx) for suite in suites for idx, _ in enumerate(cls[suite])]

seen = set()
bad = 0
for path in glob.glob(os.path.join(out, "results_partial_*.json")) + glob.glob(os.path.join(out, "results_*.json")):
    if os.path.basename(path).startswith("summary_"):
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        bad += 1
        continue
    for row in data.get("by_task", []):
        suite = row.get("suite")
        task_id = row.get("task_id")
        trials = int(row.get("trials", 0) or 0)
        if suite is not None and task_id is not None and trials >= 1:
            seen.add((suite, int(task_id)))

print(len(expected), len(seen), len(expected) - len(seen), bad)
PYEOF
}

round=1
echo "[plus-low-boost] start gpus=$GPU_LIST nproc=$NPROC max_pairs=$MAX_PAIRS_PER_PROC main_screen=$MAIN_PLUS_SCREEN"
while true; do
  OUT="$(cat "$CURRENT_OUT_FILE" 2>/dev/null || true)"
  if [ -z "$OUT" ]; then
    echo "[plus-low-boost] ERROR: empty current output file: $CURRENT_OUT_FILE" >&2
    exit 2
  fi
  STEP="$(parse_step "$OUT")"
  STEP="${STEP:-14800}"

  read -r expected seen missing bad < <(plus_missing_stats "$OUT")
  echo "[plus-low-boost] $(date -u '+%Y-%m-%dT%H:%M:%SZ') round=$round step=$STEP seen=$seen/$expected missing=$missing bad=$bad out=$OUT"

  if [ "$missing" -le 0 ] && [ "$bad" -eq 0 ]; then
    if ! screen -ls | grep -q "$MAIN_PLUS_SCREEN"; then
      echo "[plus-low-boost] no missing tasks and main plus screen is gone; exiting"
      exit 0
    fi
    sleep "$IDLE_SLEEP"
    continue
  fi

  tag="plus_lowboost_step${STEP}_r${round}_n${NPROC}_m${MAX_PAIRS_PER_PROC}_$(date +%Y%m%d_%H%M%S)"
  OUT="$OUT" STEP="$STEP" NIS="$NIS" NPROC="$NPROC" GPU_LIST="$GPU_LIST" \
    TAG_PREFIX="$tag" REVERSE_MISSING=0 \
    PLUS_SET_LIBERO_CONFIG_PATH=0 SET_MUJOCO_EGL_DEVICE_ID=0 \
    LAUNCH_STAGGER_SECONDS="$LAUNCH_STAGGER_SECONDS" \
    MAX_PAIRS_PER_PROC="$MAX_PAIRS_PER_PROC" \
    ACTION_HIDDEN_DIM="$ACTION_HIDDEN_DIM" ACTION_FFN_DIM="$ACTION_FFN_DIM" \
    ACTION_ATTENTION_HEAD_DIM="$ACTION_ATTENTION_HEAD_DIM" \
    bash scripts/supplement_plus_missing.sh

  round=$((round + 1))
  sleep "$SLEEP_BETWEEN_ROUNDS"
done
