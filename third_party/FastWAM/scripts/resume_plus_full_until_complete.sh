#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"
OUT="${OUT:-$(cat /data/users/junjie/current_eval_plus_out.txt 2>/dev/null || true)}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data/users/junjie/LIBERO-plus}"
EXPECTED="${EXPECTED:-10030}"
STEP="${STEP:-14800}"
NIS="${NIS:-10}"
NPROC="${NPROC:-40}"
GPU_LIST="${GPU_LIST:-0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7}"
MAX_PAIRS_PER_PROC="${MAX_PAIRS_PER_PROC:-8}"
MAX_ROUNDS="${MAX_ROUNDS:-80}"
LOOP_SLEEP_SECONDS="${LOOP_SLEEP_SECONDS:-5}"
TAG_BASE="${TAG_BASE:-plus_full_resume_$(date +%Y%m%d_%H%M%S)}"

if [ -z "$OUT" ]; then
  echo "[plus-full-loop] ERROR: OUT is empty" >&2
  exit 2
fi

cd "$REPO" || exit 2
echo "$OUT" > /data/users/junjie/current_eval_plus_out.txt

plus_status() {
  "$PY" - "$OUT" "$LIBERO_PLUS_ROOT" "$EXPECTED" <<'PYEOF'
import glob
import json
import os
import sys

out, root, expected_arg = sys.argv[1], sys.argv[2], int(sys.argv[3])
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
cls = json.load(open(os.path.join(root, "libero/libero/benchmark/task_classification.json"), encoding="utf-8"))

expected = []
cat_map = {}
for suite in suites:
    for task_id, entry in enumerate(cls[suite]):
        expected.append((suite, task_id))
        cat_map[(suite, task_id)] = entry.get("category")
expected_set = set(expected)

seen = {}
bad = []
paths = sorted(glob.glob(os.path.join(out, "results_*.json")) + glob.glob(os.path.join(out, "results_partial_*.json")))
for path in paths:
    if os.path.basename(path).startswith("summary_"):
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        bad.append({"file": os.path.basename(path), "error": str(exc)})
        continue
    if not isinstance(data, dict):
        continue
    for row in data.get("by_task", []) or []:
        try:
            key = (row.get("suite"), int(row.get("task_id")))
        except Exception:
            continue
        if key in expected_set and key not in seen and int(row.get("trials", 0) or 0) >= 1:
            seen[key] = row

missing = [pair for pair in expected if pair not in seen]
by_suite, by_cat = {}, {}
tot_s = tot_t = 0
for suite, task_id in expected:
    row = seen.get((suite, task_id))
    if not row:
        continue
    successes = int(row.get("successes", 0) or 0)
    trials = int(row.get("trials", 0) or 0)
    cat = row.get("category") or cat_map.get((suite, task_id))
    bs = by_suite.setdefault(suite, {"successes": 0, "trials": 0})
    bc = by_cat.setdefault(cat, {"successes": 0, "trials": 0})
    bs["successes"] += successes
    bs["trials"] += trials
    bc["successes"] += successes
    bc["trials"] += trials
    tot_s += successes
    tot_t += trials

for bucket in (by_suite, by_cat):
    for item in bucket.values():
        item["rate"] = item["successes"] / max(item["trials"], 1)

complete = (not missing) and len(seen) >= min(expected_arg, len(expected)) and not bad
summary = {
    "complete": bool(complete),
    "tasks": len(seen),
    "expected_tasks": len(expected),
    "missing_tasks": len(missing),
    "bad_files": bad,
    "by_category": by_cat,
    "by_suite": by_suite,
    "overall": {"successes": tot_s, "trials": tot_t, "rate": tot_s / max(tot_t, 1)},
}
suffix = "" if complete else ".live"
json.dump(summary, open(os.path.join(out, f"summary_aggregate{suffix}.json"), "w", encoding="utf-8"), indent=2)

lines = [
    "tasks aggregated: %d/%d complete=%s missing=%d bad_files=%d" % (
        len(seen), len(expected), complete, len(missing), len(bad)
    ),
    "",
    "== per dimension ==",
]
for cat in sorted(by_cat, key=lambda k: (k is None, k)):
    item = by_cat[cat]
    lines.append("  %-22s %5d/%5d = %5.1f%%" % (cat, item["successes"], item["trials"], 100 * item["rate"]))
lines += ["", "== per suite =="]
for suite in suites:
    item = by_suite.get(suite, {"successes": 0, "trials": 0, "rate": 0.0})
    lines.append("  %-16s %5d/%5d = %5.1f%%" % (suite, item["successes"], item["trials"], 100 * item["rate"]))
lines.append("")
lines.append("OVERALL %d/%d = %.2f%%" % (tot_s, tot_t, 100 * summary["overall"]["rate"]))
open(os.path.join(out, f"summary_aggregate{suffix}.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

with open(os.path.join(out, "missing_plus_pairs.live.txt"), "w", encoding="utf-8") as f:
    for suite, task_id in missing:
        f.write(f"{suite}:{task_id}\n")

print("%d %d %d %d %d %.6f" % (1 if complete else 0, len(seen), len(expected), len(missing), tot_s, tot_s / max(tot_t, 1)))
PYEOF
}

round=1
while [ "$round" -le "$MAX_ROUNDS" ]; do
  read -r complete seen expected missing successes rate < <(plus_status)
  echo "[plus-full-loop] $(date -u '+%Y-%m-%dT%H:%M:%SZ') round=$round complete=$complete seen=$seen/$expected missing=$missing successes=$successes rate=$rate out=$OUT"
  if [ "$complete" = "1" ]; then
    echo "[plus-full-loop] complete out=$OUT"
    exit 0
  fi

  tag="${TAG_BASE}_r${round}_$(date +%Y%m%d_%H%M%S)"
  echo "[plus-full-loop] launch supplement tag=$tag nproc=$NPROC max_pairs_per_proc=$MAX_PAIRS_PER_PROC gpus=$GPU_LIST"
  OUT="$OUT" STEP="$STEP" NIS="$NIS" NPROC="$NPROC" GPU_LIST="$GPU_LIST" \
    MAX_PAIRS_PER_PROC="$MAX_PAIRS_PER_PROC" TAG_PREFIX="$tag" \
    SET_MUJOCO_EGL_DEVICE_ID="${SET_MUJOCO_EGL_DEVICE_ID:-1}" \
    LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-1}" \
    EXCL="${EXCL:-}" INCLUDE_SUITES="${INCLUDE_SUITES:-}" EXCLUDE_SUITES="${EXCLUDE_SUITES:-}" \
    bash scripts/supplement_plus_missing_active_aware.sh || true

  read -r complete_after seen_after expected_after missing_after successes_after rate_after < <(plus_status)
  echo "[plus-full-loop] $(date -u '+%Y-%m-%dT%H:%M:%SZ') after round=$round complete=$complete_after seen=$seen_after/$expected_after missing=$missing_after successes=$successes_after rate=$rate_after"
  if [ "$complete_after" = "1" ]; then
    echo "[plus-full-loop] complete out=$OUT"
    exit 0
  fi
  round=$((round + 1))
  sleep "$LOOP_SLEEP_SECONDS"
done

echo "[plus-full-loop] max rounds reached; check $OUT/summary_aggregate.live.txt" >&2
exit 1
