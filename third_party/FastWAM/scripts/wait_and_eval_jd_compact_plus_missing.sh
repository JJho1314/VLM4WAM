#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data/users/junjie/LIBERO-plus}"
LIBERO_PLUS_CONFIG="${LIBERO_PLUS_CONFIG:-/data/users/junjie/.libero_plus}"

STEPS="${STEPS:-14800 14600}"
NIS="${NIS:-10}"
EXPECTED_PLUS_TASKS="${EXPECTED_PLUS_TASKS:-10030}"
MIN_FREE_GPUS="${MIN_FREE_GPUS:-8}"
FREE_MEM_MIB="${FREE_MEM_MIB:-1000}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
PROCS_PER_GPU="${PROCS_PER_GPU:-4}"
MAX_PAIRS_PER_PROC="${MAX_PAIRS_PER_PROC:-8}"
MAX_ROUNDS_PER_STEP="${MAX_ROUNDS_PER_STEP:-600}"
SLEEP_BETWEEN_ROUNDS="${SLEEP_BETWEEN_ROUNDS:-5}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-1}"
GPU_ALLOWLIST="${GPU_ALLOWLIST:-0,1,2,3,4,5,6,7}"
TAG_BASE="${TAG_BASE:-jd_compact_plus_wait_$(date +%Y%m%d_%H%M%S)}"
OUT_PREFIX="${OUT_PREFIX:-waitqueue}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

for d in "$VENV"/lib/python3.10/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export MAGICK_HOME="${MAGICK_HOME:-/data/users/junjie/im_env}"
export LD_LIBRARY_PATH="/data/users/junjie/im_env/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cd "$REPO" || exit 2
source "$TRAIN_ENV"

mkdir -p "$REPO/evaluate_results/auto_eval_logs"

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >&2
}

normalize_steps() {
  echo "$STEPS" | tr ',' ' '
}

select_free_gpus() {
  "$PY" - "$FREE_MEM_MIB" "$GPU_ALLOWLIST" <<'PYEOF'
import subprocess
import sys

free_mem = int(sys.argv[1])
allow = {x.strip() for x in sys.argv[2].split(",") if x.strip()}
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
free = []
for line in out.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        continue
    idx, used, total, util = parts[:4]
    if allow and idx not in allow:
        continue
    if int(used) < free_mem:
        free.append(idx)
print(",".join(free))
PYEOF
}

wait_for_free_gpus() {
  local free_csv free_count
  while true; do
    free_csv="$(select_free_gpus || true)"
    if [ -n "$free_csv" ]; then
      free_count="$(echo "$free_csv" | awk -F, '{print NF}')"
    else
      free_count=0
    fi
    if [ "$free_count" -ge "$MIN_FREE_GPUS" ]; then
      log "free GPUs ready: $free_csv"
      echo "$free_csv"
      return 0
    fi
    log "waiting for GPUs: free=$free_count/$MIN_FREE_GPUS allow=$GPU_ALLOWLIST mem_threshold=${FREE_MEM_MIB}MiB free_list=${free_csv:-none}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2 || true
    sleep "$WAIT_SECONDS"
  done
}

repeat_gpu_list() {
  local csv="$1"
  "$PY" - "$csv" "$PROCS_PER_GPU" <<'PYEOF'
import sys
gpus = [g.strip() for g in sys.argv[1].split(",") if g.strip()]
repeat = int(sys.argv[2])
items = []
for gpu in gpus:
    items.extend([gpu] * repeat)
print(",".join(items))
PYEOF
}

best_existing_plus_out() {
  local step="$1"
  "$PY" - "$REPO/evaluate_results" "$RUN_ID" "$step" <<'PYEOF'
import glob
import json
import os
import sys

base, run_id, step_s = sys.argv[1:]
step = int(step_s)
pattern = os.path.join(base, f"{run_id}_step{step:06d}_libero_plus*")
best = None

def count_seen(out):
    seen = set()
    bad = 0
    for path in glob.glob(os.path.join(out, "results_*.json")) + glob.glob(os.path.join(out, "results_partial_*.json")):
        if os.path.basename(path).startswith("summary_"):
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            bad += 1
            continue
        for row in data.get("by_task", []) or []:
            try:
                key = (row.get("suite"), int(row.get("task_id")))
            except Exception:
                continue
            if int(row.get("trials", 0) or 0) >= 1:
                seen.add(key)
    return len(seen), bad

for out in glob.glob(pattern):
    if not os.path.isdir(out):
        continue
    seen, bad = count_seen(out)
    score = (seen, -bad, os.path.getmtime(out))
    if best is None or score > best[0]:
        best = (score, out)
print(best[1] if best else "")
PYEOF
}

make_new_plus_out() {
  local step="$1"
  local out="$REPO/evaluate_results/${RUN_ID}_step$(printf '%06d' "$step")_libero_plus_full10030_${OUT_PREFIX}_${TAG_BASE}"
  mkdir -p "$out"
  echo "$out"
}

aggregate_plus() {
  local out="$1"
  "$PY" - "$out" "$LIBERO_PLUS_ROOT" "$EXPECTED_PLUS_TASKS" <<'PYEOF'
import glob
import json
import os
import sys

out, root, expected_s = sys.argv[1], sys.argv[2], sys.argv[3]
expected_n = int(expected_s)
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
for path in sorted(glob.glob(os.path.join(out, "results_*.json")) + glob.glob(os.path.join(out, "results_partial_*.json"))):
    if os.path.basename(path).startswith("summary_"):
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        bad.append({"file": os.path.basename(path), "error": str(exc)})
        continue
    for row in data.get("by_task", []) or []:
        try:
            key = (row.get("suite"), int(row.get("task_id")))
        except Exception:
            continue
        if key in expected_set and key not in seen and int(row.get("trials", 0) or 0) >= 1:
            seen[key] = row

missing = [f"{suite}:{task_id}" for suite, task_id in expected if (suite, task_id) not in seen]
by_suite, by_cat = {}, {}
tot_s = 0
tot_t = 0
for suite, task_id in expected:
    row = seen.get((suite, task_id))
    if not row:
        continue
    s = int(row.get("successes", 0) or 0)
    t = int(row.get("trials", 0) or 0)
    cat = row.get("category") or cat_map.get((suite, task_id)) or "unknown"
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

complete = (len(seen) >= expected_n and tot_t >= expected_n and not missing and not bad)
summary = {
    "complete": complete,
    "tasks": len(seen),
    "expected_tasks": len(expected),
    "missing_tasks": len(missing),
    "bad_files": bad,
    "by_category": by_cat,
    "by_suite": by_suite,
    "overall": {"successes": tot_s, "trials": tot_t, "rate": tot_s / max(tot_t, 1)},
}
json.dump(summary, open(os.path.join(out, "summary_aggregate.json"), "w", encoding="utf-8"), indent=2)
with open(os.path.join(out, "missing_plus_pairs.txt"), "w", encoding="utf-8") as f:
    for pair in missing:
        f.write(pair + "\n")

lines = [
    "tasks aggregated: %d/%d complete=%s missing=%d bad_files=%d"
    % (len(seen), len(expected), complete, len(missing), len(bad)),
    "",
    "== per dimension ==",
]
for cat in sorted(by_cat):
    item = by_cat[cat]
    lines.append("  %-22s %5d/%5d = %5.1f%%" % (cat, item["successes"], item["trials"], 100 * item["rate"]))
lines += ["", "== per suite =="]
for suite in sorted(by_suite):
    item = by_suite[suite]
    lines.append("  %-16s %5d/%5d = %5.1f%%" % (suite, item["successes"], item["trials"], 100 * item["rate"]))
lines.append("")
lines.append("OVERALL %d/%d = %.2f%%" % (tot_s, tot_t, 100 * summary["overall"]["rate"]))
text = "\n".join(lines)
open(os.path.join(out, "summary_aggregate.txt"), "w", encoding="utf-8").write(text + "\n")
print(text)
PYEOF
}

is_plus_complete() {
  local out="$1"
  "$PY" - "$out/summary_aggregate.json" <<'PYEOF'
import json
import sys
try:
    s = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    print("0")
    raise SystemExit
print("1" if s.get("complete") else "0")
PYEOF
}

eval_plus_step() {
  local step="$1"
  local out
  out="$(best_existing_plus_out "$step")"
  if [ -z "$out" ]; then
    out="$(make_new_plus_out "$step")"
  else
    mkdir -p "$out"
  fi
  local main_log="$out/waitqueue_eval.log"
  echo "$out" > /data/users/junjie/current_eval_plus_out.txt
  echo "$main_log" > /data/users/junjie/current_eval_plus_log.txt

  log "step=$step plus out=$out"
  aggregate_plus "$out" | tee -a "$main_log"
  if [ "$(is_plus_complete "$out")" = "1" ]; then
    log "step=$step already complete"
    return 0
  fi

  local round free_csv gpu_list nproc tag
  for round in $(seq 1 "$MAX_ROUNDS_PER_STEP"); do
    free_csv="$(wait_for_free_gpus)"
    gpu_list="$(repeat_gpu_list "$free_csv")"
    nproc="$(echo "$gpu_list" | awk -F, '{print NF}')"
    tag="${TAG_BASE}_step${step}_r${round}_n${nproc}"
    log "step=$step round=$round launching supplement nproc=$nproc gpu_list=$gpu_list out=$out"
    OUT="$out" STEP="$step" NIS="$NIS" NPROC="$nproc" GPU_LIST="$gpu_list" \
      TAG_PREFIX="$tag" REVERSE_MISSING=1 MAX_PAIRS_PER_PROC="$MAX_PAIRS_PER_PROC" \
      PLUS_SET_LIBERO_CONFIG_PATH=1 SET_MUJOCO_EGL_DEVICE_ID=1 EXCL="" \
      LAUNCH_STAGGER_SECONDS="$LAUNCH_STAGGER_SECONDS" \
      ACTION_HIDDEN_DIM="$ACTION_HIDDEN_DIM" ACTION_FFN_DIM="$ACTION_FFN_DIM" \
      ACTION_ATTENTION_HEAD_DIM="$ACTION_ATTENTION_HEAD_DIM" \
      bash scripts/supplement_plus_missing.sh 2>&1 | tee -a "$main_log"

    aggregate_plus "$out" | tee -a "$main_log"
    if [ "$(is_plus_complete "$out")" = "1" ]; then
      log "step=$step plus complete"
      return 0
    fi
    sleep "$SLEEP_BETWEEN_ROUNDS"
  done

  log "ERROR: step=$step plus incomplete after $MAX_ROUNDS_PER_STEP rounds"
  return 3
}

log "run_id=$RUN_ID"
log "run_dir=$RUN_DIR"
log "steps=$(normalize_steps) nis=$NIS expected_plus=$EXPECTED_PLUS_TASKS min_free_gpus=$MIN_FREE_GPUS procs_per_gpu=$PROCS_PER_GPU max_pairs_per_proc=$MAX_PAIRS_PER_PROC"
log "full categories enabled: EXCL is empty, Sensor Noise included"

rc=0
for step in $(normalize_steps); do
  if ! eval_plus_step "$step"; then
    rc=1
    break
  fi
done

if [ "$rc" -eq 0 ]; then
  log "all queued plus evals complete"
else
  log "queued plus evals stopped with rc=$rc"
fi
exit "$rc"
