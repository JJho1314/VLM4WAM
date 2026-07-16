#!/usr/bin/env bash
set -u

OUT="${OUT:-$(cat /data/users/junjie/current_eval_plus_out.txt 2>/dev/null || true)}"
EXPECTED="${EXPECTED:-10030}"
INTERVAL="${INTERVAL:-300}"
KILL_ON_COMPLETE="${KILL_ON_COMPLETE:-0}"
MAX_CHECKS="${MAX_CHECKS:-0}"

if [ -z "$OUT" ]; then
  echo "[plus-watch] ERROR: OUT is empty" >&2
  exit 2
fi

checks=0
while true; do
  read -r seen successes trials rate < <(
    python3 - "$OUT" <<'PYEOF'
import glob
import json
import os
import sys

out = sys.argv[1]
seen = {}
for path in sorted(glob.glob(os.path.join(out, "results_*.json")) + glob.glob(os.path.join(out, "results_partial_*.json"))):
    if os.path.basename(path).startswith("summary_"):
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    for row in data.get("by_task", []):
        key = (row.get("suite"), int(row.get("task_id")))
        if key not in seen and int(row.get("trials", 0)) >= 1:
            seen[key] = row
successes = sum(int(row.get("successes", 0)) for row in seen.values())
trials = sum(int(row.get("trials", 0)) for row in seen.values())
rate = successes / max(trials, 1)
print(len(seen), successes, trials, "%.6f" % rate)
PYEOF
  )
  echo "[plus-watch] $(date -u '+%Y-%m-%dT%H:%M:%SZ') seen=$seen/$EXPECTED successes=$successes/$trials rate=$rate out=$OUT"

  if [ "$seen" -ge "$EXPECTED" ]; then
    if [ "$KILL_ON_COMPLETE" = "1" ]; then
      python3 - "$OUT" <<'PYEOF'
import os
import signal
import subprocess
import sys

out = sys.argv[1]
ps = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
pids = []
for line in ps.splitlines():
    line = line.strip()
    if not line:
        continue
    pid_s, _, args = line.partition(" ")
    if "cosmos_eval_libero_plus.py" in args and out in args:
        pids.append(int(pid_s))
for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
print("[plus-watch] terminated plus pids:", " ".join(map(str, pids)) if pids else "none")
PYEOF
    fi
    echo "[plus-watch] complete"
    exit 0
  fi

  checks=$((checks + 1))
  if [ "$MAX_CHECKS" -gt 0 ] && [ "$checks" -ge "$MAX_CHECKS" ]; then
    echo "[plus-watch] max checks reached"
    exit 0
  fi
  sleep "$INTERVAL"
done
