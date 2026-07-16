#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
OUT="${OUT:-$(cat /data/users/junjie/current_eval_plus_out.txt 2>/dev/null || true)}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data/users/junjie/LIBERO-plus}"
EXPECTED="${EXPECTED:-10030}"
INTERVAL="${INTERVAL:-60}"
KILL_ON_COMPLETE="${KILL_ON_COMPLETE:-1}"

if [ -z "$OUT" ]; then
  echo "[plus-finalize] ERROR: OUT is empty" >&2
  exit 2
fi

cd "$REPO" || exit 2

while true; do
  read -r complete seen successes trials rate < <(python3 - "$OUT" "$LIBERO_PLUS_ROOT" "$EXPECTED" <<'PYEOF'
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

seen = {}
bad = []
for path in sorted(glob.glob(os.path.join(out, "results_*.json")) + glob.glob(os.path.join(out, "results_partial_*.json"))):
    if os.path.basename(path).startswith("summary_"):
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        bad.append((os.path.basename(path), str(exc)))
        continue
    if not isinstance(data, dict):
        continue
    for row in data.get("by_task", []):
        key = (row.get("suite"), int(row.get("task_id")))
        if None not in key and key not in seen and int(row.get("trials", 0) or 0) >= 1:
            seen[key] = row

missing = [f"{suite}:{task_id}" for suite, task_id in expected if (suite, task_id) not in seen]
by_suite, by_cat = {}, {}
tot_s = tot_t = 0
for suite, task_id in expected:
    row = seen.get((suite, task_id))
    if not row:
        continue
    s, t = int(row.get("successes", 0) or 0), int(row.get("trials", 0) or 0)
    cat = row.get("category") or cat_map[(suite, task_id)]
    bs = by_suite.setdefault(suite, {"successes": 0, "trials": 0})
    bc = by_cat.setdefault(cat, {"successes": 0, "trials": 0})
    bs["successes"] += s
    bs["trials"] += t
    bc["successes"] += s
    bc["trials"] += t
    tot_s += s
    tot_t += t
for bucket in (by_suite, by_cat):
    for item in bucket.values():
        item["rate"] = item["successes"] / max(item["trials"], 1)

complete = (not missing) and len(seen) >= expected_arg
if complete:
    summary = {
        "complete": True,
        "tasks": len(seen),
        "expected_tasks": len(expected),
        "missing_tasks": 0,
        "bad_files": bad,
        "by_category": by_cat,
        "by_suite": by_suite,
        "overall": {"successes": tot_s, "trials": tot_t, "rate": tot_s / max(tot_t, 1)},
    }
    json.dump(summary, open(os.path.join(out, "summary_aggregate.json"), "w", encoding="utf-8"), indent=2)
    lines = ["tasks aggregated: %d/%d complete=True missing=0" % (len(seen), len(expected)), "", "== per dimension =="]
    for cat in sorted(by_cat):
        item = by_cat[cat]
        lines.append("  %-22s %5d/%5d = %5.1f%%" % (cat, item["successes"], item["trials"], 100 * item["rate"]))
    lines += ["", "== per suite =="]
    for suite in sorted(by_suite):
        item = by_suite[suite]
        lines.append("  %-16s %5d/%5d = %5.1f%%" % (suite, item["successes"], item["trials"], 100 * item["rate"]))
    lines.append("")
    lines.append("OVERALL %d/%d = %.2f%%" % (tot_s, tot_t, 100 * summary["overall"]["rate"]))
    open(os.path.join(out, "summary_aggregate.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    open(os.path.join(out, "missing_plus_pairs.txt"), "w", encoding="utf-8").write("")
else:
    with open(os.path.join(out, "missing_plus_pairs.live.txt"), "w", encoding="utf-8") as f:
        for pair in missing:
            f.write(pair + "\n")

rate = tot_s / max(tot_t, 1)
print(int(complete), len(seen), tot_s, tot_t, "%.6f" % rate)
PYEOF
)
  echo "[plus-finalize] $(date -u '+%Y-%m-%dT%H:%M:%SZ') complete=$complete seen=$seen/$EXPECTED successes=$successes trials=$trials rate=$rate out=$OUT"
  if [ "$complete" = "1" ]; then
    if [ "$KILL_ON_COMPLETE" = "1" ]; then
      pkill -f "cosmos_eval_libero_plus.py.*--out_dir $OUT" || true
    fi
    echo "[plus-finalize] done"
    exit 0
  fi
  sleep "$INTERVAL"
done
