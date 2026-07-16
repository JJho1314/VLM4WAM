#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"

FIRST_OUT="${FIRST_OUT:-$(cat /data/users/junjie/current_eval_plus_out.txt 2>/dev/null || true)}"
FIRST_STEP="${FIRST_STEP:-14800}"
START_STEP_AFTER="${START_STEP_AFTER:-14600}"
MIN_STEP="${MIN_STEP:-11200}"
STEP_INTERVAL="${STEP_INTERVAL:-200}"
NIS="${NIS:-10}"
EXPECTED="${EXPECTED:-10030}"
SUPP_GPU_LIST="${SUPP_GPU_LIST:-0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,4,5,6,7}"
SUPP_NPROC="${SUPP_NPROC:-29}"
SUPP_MAX_PAIRS_PER_PROC="${SUPP_MAX_PAIRS_PER_PROC:-8}"
SUPP_ROUNDS="${SUPP_ROUNDS:-500}"
SUPP_REVERSE_MISSING="${SUPP_REVERSE_MISSING:-1}"
SUPP_SET_LIBERO_CONFIG_PATH="${SUPP_SET_LIBERO_CONFIG_PATH:-0}"
SUPP_SET_MUJOCO_EGL_DEVICE_ID="${SUPP_SET_MUJOCO_EGL_DEVICE_ID:-0}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-1}"
SLEEP_BETWEEN_ROUNDS="${SLEEP_BETWEEN_ROUNDS:-2}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

if [ -z "$FIRST_OUT" ]; then
  echo "[plus-chunked] ERROR: FIRST_OUT is empty" >&2
  exit 2
fi

cd "$REPO" || exit 2
source "$TRAIN_ENV"

parse_plus() {
  local out="$1"
  "$PY" - "$out" <<'PYEOF'
import glob
import json
import os
import sys

out = sys.argv[1]
seen = {}
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
        key = (row.get("suite"), row.get("task_id"))
        if None not in key and key not in seen and int(row.get("trials", 0) or 0) >= 1:
            seen[key] = row
succ = sum(int(row.get("successes", 0) or 0) for row in seen.values())
trials = sum(int(row.get("trials", 0) or 0) for row in seen.values())
print(len(seen), succ, trials, bad)
PYEOF
}

write_plus_summary() {
  local out="$1"
  "$PY" - "$out" "$EXPECTED" <<'PYEOF'
import glob
import json
import os
import sys
from collections import defaultdict

out, expected = sys.argv[1], int(sys.argv[2])
seen = {}
bad = []
for path in sorted(glob.glob(os.path.join(out, "results_partial_*.json")) + glob.glob(os.path.join(out, "results_*.json"))):
    if os.path.basename(path).startswith("summary_"):
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        bad.append({"file": os.path.basename(path), "error": str(exc)})
        continue
    for row in data.get("by_task", []):
        key = (row.get("suite"), row.get("task_id"))
        if None not in key and key not in seen and int(row.get("trials", 0) or 0) >= 1:
            seen[key] = row

by_suite = defaultdict(lambda: {"successes": 0, "trials": 0})
by_category = defaultdict(lambda: {"successes": 0, "trials": 0})
tot_s = 0
tot_t = 0
for row in seen.values():
    s = int(row.get("successes", 0) or 0)
    t = int(row.get("trials", 0) or 0)
    tot_s += s
    tot_t += t
    by_suite[row.get("suite")]["successes"] += s
    by_suite[row.get("suite")]["trials"] += t
    by_category[row.get("category") or "unknown"]["successes"] += s
    by_category[row.get("category") or "unknown"]["trials"] += t

for group in (by_suite, by_category):
    for item in group.values():
        item["rate"] = item["successes"] / max(item["trials"], 1)

summary = {
    "complete": len(seen) >= expected and tot_t >= expected and not bad,
    "tasks": len(seen),
    "expected_tasks": expected,
    "bad_files": bad,
    "by_suite": dict(sorted(by_suite.items())),
    "by_category": dict(sorted(by_category.items())),
    "overall": {"successes": tot_s, "trials": tot_t, "rate": tot_s / max(tot_t, 1)},
}
json.dump(summary, open(os.path.join(out, "summary_aggregate.json"), "w", encoding="utf-8"), indent=2)
lines = [f"tasks aggregated: {len(seen)} complete={summary['complete']} bad_files={len(bad)}", "", "== per dimension =="]
for cat, item in sorted(by_category.items()):
    lines.append("  %-22s %5d/%5d = %5.1f%%" % (cat, item["successes"], item["trials"], 100 * item["rate"]))
lines.extend(["", "== per suite =="])
for suite, item in sorted(by_suite.items()):
    lines.append("  %-16s %5d/%5d = %5.1f%%" % (suite, item["successes"], item["trials"], 100 * item["rate"]))
lines.append("")
lines.append("OVERALL %d/%d = %.2f%%" % (tot_s, tot_t, 100 * summary["overall"]["rate"]))
text = "\n".join(lines)
open(os.path.join(out, "summary_aggregate.txt"), "w", encoding="utf-8").write(text + "\n")
print(text)
PYEOF
}

supplement_until_complete() {
  local out="$1"
  local step="$2"
  local round
  for round in $(seq 1 "$SUPP_ROUNDS"); do
    read -r tasks succ trials bad < <(parse_plus "$out")
    echo "[plus-chunked] $(date -u '+%Y-%m-%dT%H:%M:%SZ') step=$step round=$round before tasks=$tasks succ=$succ trials=$trials bad=$bad out=$out"
    if [ "$tasks" -ge "$EXPECTED" ] && [ "$trials" -ge "$EXPECTED" ] && [ "$bad" -eq 0 ]; then
      write_plus_summary "$out"
      return 0
    fi
    local tag="plus_chunked_step${step}_r${round}_n${SUPP_NPROC}_m${SUPP_MAX_PAIRS_PER_PROC}_$(date +%Y%m%d_%H%M%S)"
    OUT="$out" STEP="$step" NIS="$NIS" NPROC="$SUPP_NPROC" GPU_LIST="$SUPP_GPU_LIST" \
      TAG_PREFIX="$tag" REVERSE_MISSING="$SUPP_REVERSE_MISSING" \
      PLUS_SET_LIBERO_CONFIG_PATH="$SUPP_SET_LIBERO_CONFIG_PATH" \
      SET_MUJOCO_EGL_DEVICE_ID="$SUPP_SET_MUJOCO_EGL_DEVICE_ID" \
      LAUNCH_STAGGER_SECONDS="$LAUNCH_STAGGER_SECONDS" \
      MAX_PAIRS_PER_PROC="$SUPP_MAX_PAIRS_PER_PROC" \
      ACTION_HIDDEN_DIM="$ACTION_HIDDEN_DIM" ACTION_FFN_DIM="$ACTION_FFN_DIM" \
      ACTION_ATTENTION_HEAD_DIM="$ACTION_ATTENTION_HEAD_DIM" \
      bash scripts/supplement_plus_missing.sh
    read -r tasks succ trials bad < <(parse_plus "$out")
    echo "[plus-chunked] $(date -u '+%Y-%m-%dT%H:%M:%SZ') step=$step round=$round after tasks=$tasks succ=$succ trials=$trials bad=$bad"
    if [ "$tasks" -ge "$EXPECTED" ] && [ "$trials" -ge "$EXPECTED" ] && [ "$bad" -eq 0 ]; then
      write_plus_summary "$out"
      return 0
    fi
    sleep "$SLEEP_BETWEEN_ROUNDS"
  done
  write_plus_summary "$out" || true
  echo "[plus-chunked] ERROR: plus incomplete after chunked supplements step=$step out=$out" >&2
  return 3
}

run_empty_step() {
  local step="$1"
  local ts="$2"
  local out="$REPO/evaluate_results/${RUN_ID}_step$(printf '%06d' "$step")_libero_plus_full10030_chunked_${ts}"
  local log="$REPO/evaluate_results/auto_eval_logs/plus_chunked_step${step}_${ts}.log"
  mkdir -p "$out"
  echo "$out" > /data/users/junjie/current_eval_plus_out.txt
  echo "$log" > /data/users/junjie/current_eval_plus_log.txt
  echo "[plus-chunked] new step=$step out=$out log=$log"
  supplement_until_complete "$out" "$step" > "$log" 2>&1
}

TS="$(date +%Y%m%d_%H%M%S)"
echo "[plus-chunked] first_step=$FIRST_STEP first_out=$FIRST_OUT next=$START_STEP_AFTER min=$MIN_STEP gpus=$SUPP_GPU_LIST nproc=$SUPP_NPROC max_pairs=$SUPP_MAX_PAIRS_PER_PROC"
supplement_until_complete "$FIRST_OUT" "$FIRST_STEP" || exit $?

step="$START_STEP_AFTER"
while [ "$step" -ge "$MIN_STEP" ]; do
  run_empty_step "$step" "$TS" || exit $?
  step=$((step - STEP_INTERVAL))
done

echo "[plus-chunked] all queued plus evals complete"
