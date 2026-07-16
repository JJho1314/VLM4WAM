#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
OUT="${OUT:?OUT is required}"
STEP="${STEP:?STEP is required}"
NIS="${NIS:-10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
STANDARD_NPROC="${STANDARD_NPROC:-80}"
STANDARD_GPU_LIST="${STANDARD_GPU_LIST:-0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7}"
STANDARD_ROUNDS="${STANDARD_ROUNDS:-80}"
TRIALS_PER_JOB="${TRIALS_PER_JOB:-1}"
PAIRS_PER_JOB="${PAIRS_PER_JOB:-1}"
STANDARD_SET_MUJOCO_EGL_DEVICE_ID="${STANDARD_SET_MUJOCO_EGL_DEVICE_ID:-1}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-0.5}"
ROUND_SLEEP_SECONDS="${ROUND_SLEEP_SECONDS:-3}"
TAG_BASE="${TAG_BASE:-standard_only_wave_$(date +%Y%m%d_%H%M%S)}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

source "$TRAIN_ENV"
cd "$REPO" || exit 2
mkdir -p "$OUT" "$REPO/evaluate_results/auto_eval_logs"
echo "$OUT" > /data/users/junjie/current_eval_standard_out.txt

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
    success = int(seen[(suite, task_id, trial)])
    by_task[(suite, task_id)]["successes"] += success
    by_task[(suite, task_id)]["trials"] += 1
    by_suite[suite]["successes"] += success
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
with open(os.path.join(out, "summary_aggregate.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
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
with open(os.path.join(out, "summary_aggregate.txt"), "w", encoding="utf-8") as f:
    f.write(text + "\n")
print(text)
PYEOF
}

round=1
while [ "$round" -le "$STANDARD_ROUNDS" ]; do
  summary_text="$(write_standard_summary "$OUT")"
  echo "$summary_text"
  units="$("$PY" - "$OUT/summary_aggregate.json" <<'PYEOF'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(int(d.get("trial_units", 0)), int(d.get("missing_trial_units", 0)), int(bool(d.get("bad_files"))))
PYEOF
)"
  read -r trial_units missing has_bad <<< "$units"
  echo "[standard-wave] $(date -u '+%Y-%m-%dT%H:%M:%SZ') step=$STEP round=$round units=$trial_units/2000 missing=$missing bad=$has_bad nproc=$STANDARD_NPROC gpus=$STANDARD_GPU_LIST"
  if [ "$missing" -le 0 ] && [ "$has_bad" -eq 0 ]; then
    echo "[standard-wave] complete step=$STEP out=$OUT"
    exit 0
  fi
  tag="${TAG_BASE}_r${round}_$(date +%Y%m%d_%H%M%S)"
  OUT="$OUT" STEP="$STEP" NIS="$NIS" NPROC="$STANDARD_NPROC" GPU_LIST="$STANDARD_GPU_LIST" \
    NUM_TRIALS="$NUM_TRIALS" TAG_PREFIX="$tag" \
    SET_MUJOCO_EGL_DEVICE_ID="$STANDARD_SET_MUJOCO_EGL_DEVICE_ID" \
    PAIRS_PER_JOB="$PAIRS_PER_JOB" TRIALS_PER_JOB="$TRIALS_PER_JOB" \
    LAUNCH_STAGGER_SECONDS="$LAUNCH_STAGGER_SECONDS" \
    ACTION_HIDDEN_DIM="$ACTION_HIDDEN_DIM" ACTION_FFN_DIM="$ACTION_FFN_DIM" \
    ACTION_ATTENTION_HEAD_DIM="$ACTION_ATTENTION_HEAD_DIM" \
    bash scripts/supplement_standard_trials_missing.sh
  round=$((round + 1))
  sleep "$ROUND_SLEEP_SECONDS"
done

write_standard_summary "$OUT"
echo "[standard-wave] ERROR: reached STANDARD_ROUNDS=$STANDARD_ROUNDS before completion" >&2
exit 3
