#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
START_STEP="${START_STEP:-14600}"
MIN_STEP="${MIN_STEP:-11200}"
STEP_INTERVAL="${STEP_INTERVAL:-200}"
NIS="${NIS:-10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
TRIALS_PER_JOB="${TRIALS_PER_JOB:-1}"
STANDARD_NPROC="${STANDARD_NPROC:-10}"
STANDARD_GPU_LIST="${STANDARD_GPU_LIST:-3,4,5,6,7}"
STANDARD_SET_MUJOCO_EGL_DEVICE_ID="${STANDARD_SET_MUJOCO_EGL_DEVICE_ID:-0}"
STANDARD_ROUNDS="${STANDARD_ROUNDS:-80}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-1}"
PAIRS_PER_JOB="${PAIRS_PER_JOB:-3}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

source "$TRAIN_ENV"
cd "$REPO" || exit 2
mkdir -p "$REPO/evaluate_results/auto_eval_logs"

parse_standard() {
  local out="$1"
  "$PY" - "$out" "$NUM_TRIALS" <<'PYEOF'
import glob
import json
import os
import sys

out, num_trials = sys.argv[1], int(sys.argv[2])
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
expected = {(suite, task_id, trial) for suite in suites for task_id in range(10) for trial in range(num_trials)}
seen = {}
bad = 0
for path in sorted(glob.glob(os.path.join(out, "results_partial_*.json")) + glob.glob(os.path.join(out, "results_*.json"))):
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
        if suite is None or task_id is None:
            continue
        trial_start = int(row.get("trial_start", 0) or 0)
        trials = int(row.get("trials", 0) or 0)
        successes = int(row.get("successes", 0) or 0)
        for off in range(trials):
            key = (suite, int(task_id), trial_start + off)
            if key in expected and key not in seen:
                # Split multi-trial rows only as a fallback. Trial-level runs have trials=1.
                seen[key] = 1 if off < successes else 0
succ = sum(seen.values())
print(len(seen), succ, len(seen), bad)
PYEOF
}

write_standard_summary() {
  local out="$1"
  "$PY" - "$out" "$NUM_TRIALS" <<'PYEOF'
import glob
import json
import os
import sys
from collections import defaultdict

out, num_trials = sys.argv[1], int(sys.argv[2])
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
expected = {(suite, task_id, trial) for suite in suites for task_id in range(10) for trial in range(num_trials)}
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
        suite = row.get("suite")
        task_id = row.get("task_id")
        if suite is None or task_id is None:
            continue
        trial_start = int(row.get("trial_start", 0) or 0)
        trials = int(row.get("trials", 0) or 0)
        successes = int(row.get("successes", 0) or 0)
        for off in range(trials):
            key = (suite, int(task_id), trial_start + off)
            if key in expected and key not in seen:
                seen[key] = 1 if off < successes else 0

by_task = defaultdict(lambda: {"successes": 0, "trials": 0})
by_suite = defaultdict(lambda: {"successes": 0, "trials": 0})
for suite, task_id, trial in sorted(seen):
    s = int(seen[(suite, task_id, trial)])
    by_task[(suite, task_id)]["successes"] += s
    by_task[(suite, task_id)]["trials"] += 1
    by_suite[suite]["successes"] += s
    by_suite[suite]["trials"] += 1

for group in (by_task, by_suite):
    for item in group.values():
        item["rate"] = item["successes"] / max(item["trials"], 1)

tot_s = sum(seen.values())
tot_t = len(seen)
summary = {
    "complete": len(seen) == len(expected) and not bad,
    "tasks": len(by_task),
    "trial_units": len(seen),
    "expected_trial_units": len(expected),
    "missing_trial_units": len(expected) - len(seen),
    "bad_files": bad,
    "by_task": {f"{suite}:{task_id}": item for (suite, task_id), item in sorted(by_task.items())},
    "by_suite": dict(sorted(by_suite.items())),
    "overall": {"successes": tot_s, "trials": tot_t, "rate": tot_s / max(tot_t, 1)},
}
json.dump(summary, open(os.path.join(out, "summary_aggregate.json"), "w", encoding="utf-8"), indent=2)
with open(os.path.join(out, "missing_standard_trials.live.txt"), "w", encoding="utf-8") as f:
    for suite, task_id, trial in sorted(expected - set(seen), key=lambda x: (x[2], x[0], x[1])):
        f.write(f"{suite}:{task_id}:{trial}\n")
lines = [
    "trial units aggregated: %d/%d complete=%s bad_files=%d" % (
        len(seen), len(expected), summary["complete"], len(bad)
    ),
    "",
    "== per suite ==",
]
for suite in suites:
    item = by_suite.get(suite, {"successes": 0, "trials": 0, "rate": 0.0})
    lines.append("  %-16s %4d/%4d = %.2f%%" % (suite, item["successes"], item["trials"], 100 * item["rate"]))
lines.append("")
lines.append("OVERALL %d/%d = %.2f%%" % (tot_s, tot_t, 100 * summary["overall"]["rate"]))
text = "\n".join(lines)
open(os.path.join(out, "summary_aggregate.txt"), "w", encoding="utf-8").write(text + "\n")
print(text)
PYEOF
}

complete_existing() {
  local step="$1"
  "$PY" - "$REPO/evaluate_results" "$RUN_ID" "$step" <<'PYEOF'
import glob
import json
import os
import sys

root, run_id, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
pattern = os.path.join(root, f"{run_id}_step{step:06d}_libero_standard50_*", "summary_aggregate.json")
for path in sorted(glob.glob(pattern), reverse=True):
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    overall = data.get("overall", {})
    trials = int(overall.get("trials", 0) or 0)
    trial_units = int(data.get("trial_units", 0) or 0)
    tasks = int(data.get("tasks", 0) or 0)
    if (tasks == 40 and trials == 2000) or (trial_units == 2000 and trials == 2000):
        print(os.path.dirname(path))
        break
PYEOF
}

TS="$(date +%Y%m%d_%H%M%S)"
echo "[standard-trials-backfill] run=$RUN_ID start=$START_STEP min=$MIN_STEP nproc=$STANDARD_NPROC gpus=$STANDARD_GPU_LIST pairs_per_job=$PAIRS_PER_JOB trials_per_job=$TRIALS_PER_JOB rounds=$STANDARD_ROUNDS"

step="$START_STEP"
while [ "$step" -ge "$MIN_STEP" ]; do
  existing="$(complete_existing "$step")"
  if [ -n "$existing" ]; then
    echo "[standard-trials-backfill] step=$step already complete: $existing"
    step=$((step - STEP_INTERVAL))
    continue
  fi

  out="$REPO/evaluate_results/${RUN_ID}_step$(printf '%06d' "$step")_libero_standard50_trials_${TS}"
  log="$REPO/evaluate_results/auto_eval_logs/standard_trials_step${step}_${TS}.log"
  mkdir -p "$out"
  echo "$out" > /data/users/junjie/current_eval_standard_out.txt
  echo "$log" > /data/users/junjie/current_eval_standard_log.txt
  echo "[standard-trials-backfill] step=$step out=$out log=$log"

  round=1
  while [ "$round" -le "$STANDARD_ROUNDS" ]; do
    read -r units succ trials bad < <(parse_standard "$out")
    echo "[standard-trials-backfill] $(date -u '+%Y-%m-%dT%H:%M:%SZ') step=$step round=$round units=$units/2000 succ=$succ trials=$trials bad=$bad" | tee -a "$log"
    if [ "$units" -ge 2000 ] && [ "$trials" -ge 2000 ] && [ "$bad" -eq 0 ]; then
      write_standard_summary "$out" | tee -a "$log"
      break
    fi
    tag="standard_trials_step${step}_r${round}_$(date +%Y%m%d_%H%M%S)"
    OUT="$out" STEP="$step" NIS="$NIS" NPROC="$STANDARD_NPROC" GPU_LIST="$STANDARD_GPU_LIST" \
      NUM_TRIALS="$NUM_TRIALS" TAG_PREFIX="$tag" \
      SET_MUJOCO_EGL_DEVICE_ID="$STANDARD_SET_MUJOCO_EGL_DEVICE_ID" \
      PAIRS_PER_JOB="$PAIRS_PER_JOB" \
      TRIALS_PER_JOB="$TRIALS_PER_JOB" \
      LAUNCH_STAGGER_SECONDS="$LAUNCH_STAGGER_SECONDS" \
      ACTION_HIDDEN_DIM="$ACTION_HIDDEN_DIM" ACTION_FFN_DIM="$ACTION_FFN_DIM" \
      ACTION_ATTENTION_HEAD_DIM="$ACTION_ATTENTION_HEAD_DIM" \
      bash scripts/supplement_standard_trials_missing.sh 2>&1 | tee -a "$log"
    round=$((round + 1))
  done

  read -r units succ trials bad < <(parse_standard "$out")
  if [ "$units" -lt 2000 ] || [ "$trials" -lt 2000 ] || [ "$bad" -ne 0 ]; then
    write_standard_summary "$out" | tee -a "$log" || true
    echo "[standard-trials-backfill] ERROR: incomplete step=$step units=$units trials=$trials bad=$bad" | tee -a "$log" >&2
    exit 3
  fi
  step=$((step - STEP_INTERVAL))
done

echo "[standard-trials-backfill] complete"
