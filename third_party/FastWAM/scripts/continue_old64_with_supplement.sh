#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"

FIRST_SCREEN="${FIRST_SCREEN:-}"
FIRST_OUT="${FIRST_OUT:-$(cat /data/users/junjie/current_eval_plus_out.txt 2>/dev/null || true)}"
FIRST_STEP="${FIRST_STEP:-14800}"
START_STEP_AFTER="${START_STEP_AFTER:-14600}"
MIN_STEP="${MIN_STEP:-11200}"
STEP_INTERVAL="${STEP_INTERVAL:-200}"
NIS="${NIS:-10}"
EXPECTED="${EXPECTED:-10030}"
WAIT_INTERVAL="${WAIT_INTERVAL:-300}"

STANDARD_NPROC="${STANDARD_NPROC:-40}"
PLUS_NPROC="${PLUS_NPROC:-64}"
SUPP_GPU_LIST="${SUPP_GPU_LIST:-0,1,2,3,4,5,6,7}"
SUPP_NPROC_SEQUENCE="${SUPP_NPROC_SEQUENCE:-16 8 4 2 1}"
SUPP_ROUNDS="${SUPP_ROUNDS:-3}"
SUPP_SET_LIBERO_CONFIG_PATH="${SUPP_SET_LIBERO_CONFIG_PATH:-0}"
SUPP_SET_MUJOCO_EGL_DEVICE_ID="${SUPP_SET_MUJOCO_EGL_DEVICE_ID:-0}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-1}"
PLUS_MAIN_MODE="${PLUS_MAIN_MODE:-old64}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

if [ -z "$FIRST_OUT" ]; then
  echo "[continue-old64-supp] ERROR: FIRST_OUT is empty" >&2
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
        if None not in key and key not in seen:
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
        if None not in key and key not in seen:
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

screen_exists() {
  local name="$1"
  [ -n "$name" ] && screen -ls | grep -q "[.]${name}[[:space:]]"
}

wait_for_main_or_complete() {
  local out="$1"
  local screen_name="$2"
  local step="$3"
  while true; do
    read -r tasks succ trials bad < <(parse_plus "$out")
    echo "[continue-old64-supp] $(date -u '+%Y-%m-%dT%H:%M:%SZ') step=$step main-wait tasks=$tasks succ=$succ trials=$trials bad=$bad out=$out"
    if [ "$tasks" -ge "$EXPECTED" ] && [ "$trials" -ge "$EXPECTED" ] && [ "$bad" -eq 0 ]; then
      write_plus_summary "$out"
      return 0
    fi
    if ! screen_exists "$screen_name"; then
      echo "[continue-old64-supp] main screen ended for step=$step; will supplement missing pairs"
      return 1
    fi
    sleep "$WAIT_INTERVAL"
  done
}

supplement_until_complete() {
  local out="$1"
  local step="$2"
  local round
  for round in $(seq 1 "$SUPP_ROUNDS"); do
    read -r tasks succ trials bad < <(parse_plus "$out")
    echo "[continue-old64-supp] $(date -u '+%Y-%m-%dT%H:%M:%SZ') step=$step supp-round=$round before tasks=$tasks succ=$succ trials=$trials bad=$bad"
    if [ "$tasks" -ge "$EXPECTED" ] && [ "$trials" -ge "$EXPECTED" ] && [ "$bad" -eq 0 ]; then
      write_plus_summary "$out"
      return 0
    fi
    local nproc
    for nproc in $SUPP_NPROC_SEQUENCE; do
      local tag
      tag="plus_supp_old64_step${step}_r${round}_n${nproc}_$(date +%Y%m%d_%H%M%S)"
      echo "[continue-old64-supp] supplement step=$step round=$round nproc=$nproc tag=$tag"
      OUT="$out" STEP="$step" NIS="$NIS" NPROC="$nproc" GPU_LIST="$SUPP_GPU_LIST" \
        TAG_PREFIX="$tag" PLUS_SET_LIBERO_CONFIG_PATH="$SUPP_SET_LIBERO_CONFIG_PATH" \
        SET_MUJOCO_EGL_DEVICE_ID="$SUPP_SET_MUJOCO_EGL_DEVICE_ID" \
        LAUNCH_STAGGER_SECONDS="$LAUNCH_STAGGER_SECONDS" \
        ACTION_HIDDEN_DIM="$ACTION_HIDDEN_DIM" ACTION_FFN_DIM="$ACTION_FFN_DIM" \
        ACTION_ATTENTION_HEAD_DIM="$ACTION_ATTENTION_HEAD_DIM" \
        bash scripts/supplement_plus_missing.sh
      read -r tasks succ trials bad < <(parse_plus "$out")
      echo "[continue-old64-supp] after supplement step=$step nproc=$nproc tasks=$tasks succ=$succ trials=$trials bad=$bad"
      if [ "$tasks" -ge "$EXPECTED" ] && [ "$trials" -ge "$EXPECTED" ] && [ "$bad" -eq 0 ]; then
        write_plus_summary "$out"
        return 0
      fi
    done
  done
  write_plus_summary "$out" || true
  echo "[continue-old64-supp] ERROR: plus incomplete after supplements step=$step out=$out" >&2
  return 3
}

run_full_step_old64() {
  local step="$1"
  local ts="$2"
  local step_log="$REPO/evaluate_results/auto_eval_logs/eval_old64_step${step}_${ts}.log"
  echo "[continue-old64-supp] starting step=$step mode=$PLUS_MAIN_MODE log=$step_log"
  if [ "$PLUS_MAIN_MODE" = "supp_only" ]; then
    STEP="$step" STANDARD_NPROC="$STANDARD_NPROC" PLUS_NPROC="$PLUS_NPROC" NIS="$NIS" \
      SKIP_PLUS=1 ACTION_HIDDEN_DIM="$ACTION_HIDDEN_DIM" ACTION_FFN_DIM="$ACTION_FFN_DIM" \
      ACTION_ATTENTION_HEAD_DIM="$ACTION_ATTENTION_HEAD_DIM" \
      bash scripts/auto_eval_joint_denoise_after_train.sh > "$step_log" 2>&1
    local plus_ts out plus_log
    plus_ts="$(date +%Y%m%d_%H%M%S)"
    out="$REPO/evaluate_results/${RUN_ID}_step$(printf '%06d' "$step")_libero_plus_full10030_supp_${plus_ts}"
    plus_log="$REPO/evaluate_results/auto_eval_logs/plus_supp_step${step}_${plus_ts}.log"
    mkdir -p "$out"
    echo "$out" > /data/users/junjie/current_eval_plus_out.txt
    echo "$plus_log" > /data/users/junjie/current_eval_plus_log.txt
    echo "[continue-old64-supp] supp_only plus_out=$out plus_log=$plus_log"
    supplement_until_complete "$out" "$step"
  else
    STEP="$step" STANDARD_NPROC="$STANDARD_NPROC" PLUS_NPROC="$PLUS_NPROC" NIS="$NIS" \
      ACTION_HIDDEN_DIM="$ACTION_HIDDEN_DIM" ACTION_FFN_DIM="$ACTION_FFN_DIM" \
      ACTION_ATTENTION_HEAD_DIM="$ACTION_ATTENTION_HEAD_DIM" \
      bash scripts/auto_eval_joint_denoise_after_train.sh > "$step_log" 2>&1
    local out
    out="$(cat /data/users/junjie/current_eval_plus_out.txt)"
    echo "[continue-old64-supp] old64 main finished step=$step plus_out=$out"
    supplement_until_complete "$out" "$step"
  fi
}

TS="$(date +%Y%m%d_%H%M%S)"
echo "[continue-old64-supp] first_screen=$FIRST_SCREEN first_out=$FIRST_OUT first_step=$FIRST_STEP"
if ! wait_for_main_or_complete "$FIRST_OUT" "$FIRST_SCREEN" "$FIRST_STEP"; then
  supplement_until_complete "$FIRST_OUT" "$FIRST_STEP" || exit $?
fi

step="$START_STEP_AFTER"
while [ "$step" -ge "$MIN_STEP" ]; do
  run_full_step_old64 "$step" "$TS" || exit $?
  step=$((step - STEP_INTERVAL))
done

echo "[continue-old64-supp] all queued old64+supplement evals complete"
