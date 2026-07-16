#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
STEP="${STEP:-15000}"
NIS="${NIS:-10}"
STANDARD_NPROC="${STANDARD_NPROC:-40}"
STANDARD_GPU_LIST="${STANDARD_GPU_LIST:-0,1,2,3,4,5,6,7}"
STANDARD_SET_MUJOCO_EGL_DEVICE_ID="${STANDARD_SET_MUJOCO_EGL_DEVICE_ID:-0}"
PLUS_NPROC="${PLUS_NPROC:-64}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data/users/junjie/LIBERO-plus}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

for d in "$VENV"/lib/python3.10/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

source "$TRAIN_ENV"
cd "$REPO"

CKPT="$RUN_DIR/checkpoints/weights/step_$(printf "%06d" "$STEP").pt"
AUTO_LOG_DIR="$REPO/evaluate_results/auto_eval_logs"
mkdir -p "$AUTO_LOG_DIR"

train_running() {
  pgrep -af "scripts/train.py" | grep -F "$RUN_ID" >/dev/null 2>&1
}

echo "[auto-eval] waiting for training checkpoint: $CKPT"
while true; do
  if [ -f "$CKPT" ] && ! train_running; then
    break
  fi
  date
  if [ -f "$CKPT" ]; then
    echo "[auto-eval] final checkpoint exists; waiting for train process to exit"
  else
    latest="$(find "$RUN_DIR/checkpoints/weights" -maxdepth 1 -name 'step_*.pt' 2>/dev/null | sort | tail -1 || true)"
    echo "[auto-eval] latest checkpoint: ${latest:-none}"
  fi
  sleep 300
done

if [ "${SKIP_STANDARD_IF_EXISTS:-0}" = "1" ]; then
  EXISTING_STD="$("$PY" - "$REPO/evaluate_results" "$RUN_ID" "$STEP" <<'PYEOF'
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
    tasks = int(data.get("tasks", 0) or 0)
    overall = data.get("overall", {})
    trials = int(overall.get("trials", 0) or 0)
    if tasks == 40 and trials == 2000:
        print(os.path.dirname(path))
        break
PYEOF
)"
  if [ -n "$EXISTING_STD" ]; then
    echo "[auto-eval] found complete LIBERO standard eval for step=$STEP; skipping: $EXISTING_STD"
    echo "$EXISTING_STD" > /data/users/junjie/current_eval_standard_out.txt
    echo "$EXISTING_STD/summary_aggregate.txt" > /data/users/junjie/current_eval_standard_log.txt
    if [ "${SKIP_PLUS:-0}" = "1" ]; then
      echo "[auto-eval] SKIP_PLUS=1; standard eval already complete"
      exit 0
    fi
  fi
fi

echo "[auto-eval] training complete; starting LIBERO standard eval"
TS="$(date +%Y%m%d_%H%M%S)"
STD_OUT="$REPO/evaluate_results/${RUN_ID}_step$(printf '%06d' "$STEP")_libero_standard50_${TS}"
STD_LOG="$STD_OUT/eval_main.log"
mkdir -p "$STD_OUT/shards"
echo "$STD_OUT" > /data/users/junjie/current_eval_standard_out.txt
echo "$STD_LOG" > /data/users/junjie/current_eval_standard_log.txt

"$PY" - "$STANDARD_NPROC" "$STD_OUT" <<'PYEOF'
import sys

nproc = int(sys.argv[1])
out = sys.argv[2]
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
pairs = [f"{suite}:{task_id}" for suite in suites for task_id in range(10)]
shards = [[] for _ in range(nproc)]
for idx, pair in enumerate(pairs):
    shards[idx % nproc].append(pair)
for idx, shard in enumerate(shards):
    with open(f"{out}/shards/shard_{idx}.txt", "w", encoding="utf-8") as f:
        f.write(",".join(shard))
print(f"standard pairs {len(pairs)} -> {nproc} shards")
PYEOF

IFS=',' read -r -a STANDARD_GPUS <<< "$STANDARD_GPU_LIST"
STANDARD_NGPU="${#STANDARD_GPUS[@]}"
if [ "$STANDARD_NGPU" -le 0 ]; then
  echo "[standard] ERROR: empty STANDARD_GPU_LIST=$STANDARD_GPU_LIST" >&2
  exit 2
fi

echo "[standard] launching $STANDARD_NPROC procs, num_trials=50, NIS=$NIS, gpus=$STANDARD_GPU_LIST" | tee "$STD_LOG"
for idx in $(seq 0 $((STANDARD_NPROC - 1))); do
  gpu="${STANDARD_GPUS[$((idx % STANDARD_NGPU))]}"
  pairs="$(cat "$STD_OUT/shards/shard_${idx}.txt" 2>/dev/null || true)"
  [ -z "$pairs" ] && continue
  (
    unset LIBERO_CONFIG_PATH
    export CUDA_VISIBLE_DEVICES="$gpu"
    if [ "$STANDARD_SET_MUJOCO_EGL_DEVICE_ID" = "1" ]; then
      export MUJOCO_EGL_DEVICE_ID="$gpu"
    else
      unset MUJOCO_EGL_DEVICE_ID
    fi
    "$PY" experiments/libero/cosmos_eval_libero.py \
      --pairs "$pairs" --tag "$idx" --num_trials 50 --num_inference_steps "$NIS" \
      --coupling mot --run_dir "$RUN_DIR" --step "$STEP" --out_dir "$STD_OUT" \
      --action_hidden_dim "$ACTION_HIDDEN_DIM" --action_ffn_dim "$ACTION_FFN_DIM" \
      --action_attention_head_dim "$ACTION_ATTENTION_HEAD_DIM" --no-save_videos
  ) > "$STD_OUT/proc${idx}_gpu${gpu}.log" 2>&1 &
  sleep 1
done
wait
echo "ALL-STANDARD-PROCS-DONE" | tee -a "$STD_LOG"

"$PY" - "$STD_OUT" <<'PYEOF' | tee "$STD_OUT/summary_aggregate.txt"
import glob
import json
import os
import sys

out = sys.argv[1]
seen = set()
by_suite = {}
tot_s = 0
tot_t = 0
for path in sorted(glob.glob(os.path.join(out, "results_*.json")) + glob.glob(os.path.join(out, "results_partial_*.json"))):
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    for row in data.get("by_task", []):
        key = (row["suite"], int(row["task_id"]))
        if key in seen:
            continue
        seen.add(key)
        s = int(row["successes"])
        t = int(row["trials"])
        item = by_suite.setdefault(row["suite"], {"successes": 0, "trials": 0})
        item["successes"] += s
        item["trials"] += t
        tot_s += s
        tot_t += t
for item in by_suite.values():
    item["rate"] = item["successes"] / max(item["trials"], 1)
summary = {
    "tasks": len(seen),
    "by_suite": by_suite,
    "overall": {"successes": tot_s, "trials": tot_t, "rate": tot_s / max(tot_t, 1)},
}
json.dump(summary, open(os.path.join(out, "summary_aggregate.json"), "w", encoding="utf-8"), indent=2)
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    item = by_suite.get(suite, {"successes": 0, "trials": 0, "rate": 0.0})
    print("[standard] %-16s %4d/%4d = %.2f%%" % (suite, item["successes"], item["trials"], 100 * item["rate"]))
print("[standard] OVERALL          %4d/%4d = %.2f%%" % (tot_s, tot_t, 100 * summary["overall"]["rate"]))
if len(seen) != 40 or tot_t != 2000:
    raise SystemExit("standard aggregation incomplete: tasks=%d trials=%d" % (len(seen), tot_t))
PYEOF

if [ "${SKIP_PLUS:-0}" = "1" ]; then
  echo "[auto-eval] SKIP_PLUS=1; standard eval complete"
  exit 0
fi

echo "[auto-eval] starting LIBERO Plus eval"
PLUS_TS="$(date +%Y%m%d_%H%M%S)"
PLUS_OUT="$REPO/evaluate_results/${RUN_ID}_step$(printf '%06d' "$STEP")_libero_plus_full10030_${PLUS_TS}"
PLUS_LOG="$PLUS_OUT/eval_main.log"
mkdir -p "$PLUS_OUT"
echo "$PLUS_OUT" > /data/users/junjie/current_eval_plus_out.txt
echo "$PLUS_LOG" > /data/users/junjie/current_eval_plus_log.txt
unset CKPT
NPROC="$PLUS_NPROC" EXCL="" CPL=mot RDIR="$RUN_DIR" STEP="$STEP" OUT="$PLUS_OUT" \
  LIBERO_PLUS_ROOT="$LIBERO_PLUS_ROOT" ACTION_HIDDEN_DIM="$ACTION_HIDDEN_DIM" \
  ACTION_FFN_DIM="$ACTION_FFN_DIM" ACTION_ATTENTION_HEAD_DIM="$ACTION_ATTENTION_HEAD_DIM" \
  bash experiments/libero/run_cosmos_eval_plus_par.sh 2>&1 | tee "$PLUS_LOG"

"$PY" experiments/libero/combine_plus.py "$PLUS_OUT" | tee "$PLUS_OUT/summary_aggregate.txt"
echo "[auto-eval] complete"
