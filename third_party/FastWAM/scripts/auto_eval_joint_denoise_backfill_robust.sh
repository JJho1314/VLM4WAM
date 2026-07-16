#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
START_STEP="${START_STEP:-14800}"
MIN_STEP="${MIN_STEP:-11200}"
STEP_INTERVAL="${STEP_INTERVAL:-200}"
STEPS="${STEPS:-}"
NIS="${NIS:-10}"
STANDARD_NPROC="${STANDARD_NPROC:-16}"
PLUS_NPROC="${PLUS_NPROC:-24}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
STANDARD_GPU_LIST="${STANDARD_GPU_LIST:-$GPU_LIST}"
PLUS_GPU_LIST="${PLUS_GPU_LIST:-$GPU_LIST}"
PLUS_LIBERO10_GPU_LIST="${PLUS_LIBERO10_GPU_LIST:-}"
PLUS_LIBERO10_NPROC="${PLUS_LIBERO10_NPROC:-24}"
STANDARD_SET_MUJOCO_EGL_DEVICE_ID="${STANDARD_SET_MUJOCO_EGL_DEVICE_ID:-0}"
PLUS_SET_MUJOCO_EGL_DEVICE_ID="${PLUS_SET_MUJOCO_EGL_DEVICE_ID:-0}"
PLUS_SET_LIBERO_CONFIG_PATH="${PLUS_SET_LIBERO_CONFIG_PATH:-1}"
SKIP_STANDARD="${SKIP_STANDARD:-0}"
SKIP_PLUS="${SKIP_PLUS:-0}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-3}"
STD_RETRIES="${STD_RETRIES:-4}"
PLUS_RETRIES="${PLUS_RETRIES:-4}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data/users/junjie/LIBERO-plus}"
LIBERO_PLUS_CONFIG="${LIBERO_PLUS_CONFIG:-/data/users/junjie/.libero_plus}"

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

source "$TRAIN_ENV"
cd "$REPO" || exit 2

RUN_TS="$(date +%Y%m%d_%H%M%S)"
AUTO_LOG_DIR="$REPO/evaluate_results/auto_eval_logs"
mkdir -p "$AUTO_LOG_DIR"

echo "[robust-eval] run=$RUN_ID"
echo "[robust-eval] run_dir=$RUN_DIR"
echo "[robust-eval] start=$START_STEP min=$MIN_STEP interval=$STEP_INTERVAL nis=$NIS gpus=$GPU_LIST"
echo "[robust-eval] standard_gpus=$STANDARD_GPU_LIST plus_gpus=$PLUS_GPU_LIST egl standard=$STANDARD_SET_MUJOCO_EGL_DEVICE_ID plus=$PLUS_SET_MUJOCO_EGL_DEVICE_ID"
echo "[robust-eval] plus_set_libero_config_path=$PLUS_SET_LIBERO_CONFIG_PATH"
echo "[robust-eval] plus_libero10_gpus=${PLUS_LIBERO10_GPU_LIST:-<same-as-plus>} plus_libero10_nproc=$PLUS_LIBERO10_NPROC"
echo "[robust-eval] standard_nproc=$STANDARD_NPROC plus_nproc=$PLUS_NPROC retries std=$STD_RETRIES plus=$PLUS_RETRIES"
echo "[robust-eval] skip_standard=$SKIP_STANDARD skip_plus=$SKIP_PLUS"

make_steps() {
  if [ -n "$STEPS" ]; then
    echo "$STEPS"
    return
  fi
  "$PY" - "$RUN_DIR/checkpoints/weights" "$START_STEP" "$MIN_STEP" "$STEP_INTERVAL" <<'PYEOF'
import os
import re
import sys

weight_dir, start, min_step, interval = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
steps = []
for name in os.listdir(weight_dir):
    m = re.fullmatch(r"step_(\d{6})\.pt", name)
    if not m:
        continue
    step = int(m.group(1))
    if min_step <= step <= start and (start - step) % interval == 0:
        steps.append(step)
print(" ".join(str(s) for s in sorted(steps, reverse=True)))
PYEOF
}

write_standard_pairs() {
  local out="$1"
  "$PY" - "$out/all_standard_pairs.txt" <<'PYEOF'
import sys

suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for suite in suites:
        for task_id in range(10):
            f.write(f"{suite}:{task_id}\n")
PYEOF
}

write_plus_pairs() {
  local out="$1"
  "$PY" - "$LIBERO_PLUS_ROOT" "$out/all_plus_pairs.txt" <<'PYEOF'
import json
import sys

root, path = sys.argv[1], sys.argv[2]
cls = json.load(open(root + "/libero/libero/benchmark/task_classification.json", encoding="utf-8"))
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
with open(path, "w", encoding="utf-8") as f:
    for suite in suites:
        for task_id, _entry in enumerate(cls[suite]):
            f.write(f"{suite}:{task_id}\n")
PYEOF
}

split_pairs() {
  local pair_file="$1"
  local nproc="$2"
  local shard_dir="$3"
  rm -rf "$shard_dir"
  mkdir -p "$shard_dir"
  "$PY" - "$pair_file" "$nproc" "$shard_dir" <<'PYEOF'
import os
import sys

pair_file, nproc, shard_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
pairs = [line.strip() for line in open(pair_file, encoding="utf-8") if line.strip()]
nproc = max(1, min(nproc, len(pairs))) if pairs else 0
shards = [[] for _ in range(nproc)]
for idx, pair in enumerate(pairs):
    shards[idx % nproc].append(pair)
for idx, shard in enumerate(shards):
    with open(os.path.join(shard_dir, f"shard_{idx}.txt"), "w", encoding="utf-8") as f:
        f.write(",".join(shard))
print(nproc)
PYEOF
}

pair_count() {
  "$PY" - "$1" <<'PYEOF'
import sys
print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()))
PYEOF
}

nproc_for_attempt() {
  local base="$1"
  local attempt="$2"
  local ngpu="$3"
  local nproc="$base"
  if [ "$attempt" -ge 1 ]; then nproc="$ngpu"; fi
  if [ "$attempt" -ge 2 ]; then nproc=$(( (ngpu + 1) / 2 )); fi
  if [ "$attempt" -ge 3 ]; then nproc=1; fi
  if [ "$nproc" -lt 1 ]; then nproc=1; fi
  echo "$nproc"
}

run_shards() {
  local mode="$1"
  local out="$2"
  local step="$3"
  local pair_file="$4"
  local base_nproc="$5"
  local attempt="$6"
  local mode_gpu_list="$GPU_LIST"
  local set_mujoco_egl="0"
  if [ "$mode" = "standard" ]; then
    mode_gpu_list="$STANDARD_GPU_LIST"
    set_mujoco_egl="$STANDARD_SET_MUJOCO_EGL_DEVICE_ID"
  elif [ "$mode" = "plus_libero10" ]; then
    mode_gpu_list="${PLUS_LIBERO10_GPU_LIST:-$PLUS_GPU_LIST}"
    set_mujoco_egl="$PLUS_SET_MUJOCO_EGL_DEVICE_ID"
  else
    mode_gpu_list="$PLUS_GPU_LIST"
    set_mujoco_egl="$PLUS_SET_MUJOCO_EGL_DEVICE_ID"
  fi
  local mode_gpus
  IFS=',' read -r -a mode_gpus <<< "$mode_gpu_list"
  local mode_ngpu="${#mode_gpus[@]}"
  if [ "$mode_ngpu" -le 0 ]; then
    echo "[$mode] ERROR: empty gpu list: $mode_gpu_list" >&2
    return 2
  fi
  local nproc
  nproc="$(nproc_for_attempt "$base_nproc" "$attempt" "$mode_ngpu")"
  local count
  count="$(pair_count "$pair_file")"
  [ "$count" -le 0 ] && return 0
  local shard_dir="$out/shards_${mode}_a${attempt}"
  nproc="$(split_pairs "$pair_file" "$nproc" "$shard_dir")"
  echo "[$mode] attempt=$attempt launching $nproc procs for $count pairs on gpus=$mode_gpu_list"

  local pids=()
  local failed=0
  local idx
  for idx in $(seq 0 $((nproc - 1))); do
    local pairs
    pairs="$(cat "$shard_dir/shard_${idx}.txt" 2>/dev/null || true)"
    [ -z "$pairs" ] && continue
    local gpu="${mode_gpus[$((idx % mode_ngpu))]}"
    local tag="${mode}_a${attempt}_${idx}"
    local log="$out/proc_${mode}_a${attempt}_${idx}_gpu${gpu}.log"
    echo "[$mode] launch idx=$idx gpu=$gpu pairs=$(awk -F, '{print NF}' <<< "$pairs") log=$log"
    if [ "$mode" = "standard" ]; then
      (
        unset LIBERO_CONFIG_PATH
        export CUDA_VISIBLE_DEVICES="$gpu"
        if [ "$set_mujoco_egl" = "1" ]; then
          export MUJOCO_EGL_DEVICE_ID="$gpu"
        else
          unset MUJOCO_EGL_DEVICE_ID
        fi
        "$PY" experiments/libero/cosmos_eval_libero.py \
          --pairs "$pairs" --tag "$tag" --num_trials 50 --num_inference_steps "$NIS" \
          --coupling mot --run_dir "$RUN_DIR" --step "$step" --out_dir "$out" \
          --action_hidden_dim "$ACTION_HIDDEN_DIM" --action_ffn_dim "$ACTION_FFN_DIM" \
          --action_attention_head_dim "$ACTION_ATTENTION_HEAD_DIM" --no-save_videos
      ) > "$log" 2>&1 &
    else
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        if [ "$set_mujoco_egl" = "1" ]; then
          export MUJOCO_EGL_DEVICE_ID="$gpu"
        else
          unset MUJOCO_EGL_DEVICE_ID
        fi
        if [ "$PLUS_SET_LIBERO_CONFIG_PATH" = "1" ]; then
          export LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG"
        else
          unset LIBERO_CONFIG_PATH
        fi
        export LIBERO_PLUS_ROOT="$LIBERO_PLUS_ROOT"
        "$PY" experiments/libero/cosmos_eval_libero_plus.py \
          --pairs "$pairs" --tag "$tag" --num_trials 1 --num_inference_steps "$NIS" \
          --coupling mot --run_dir "$RUN_DIR" --step "$step" --out_dir "$out" --exclude_categories "" \
          --action_hidden_dim "$ACTION_HIDDEN_DIM" --action_ffn_dim "$ACTION_FFN_DIM" \
          --action_attention_head_dim "$ACTION_ATTENTION_HEAD_DIM" --no-save_videos
      ) > "$log" 2>&1 &
    fi
    pids+=("$!")
    sleep "$LAUNCH_STAGGER_SECONDS"
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=$((failed + 1))
    fi
  done
  echo "[$mode] attempt=$attempt done failed_procs=$failed"
  return 0
}

split_plus_pairs_by_suite() {
  local pair_file="$1"
  local other_file="$2"
  local libero10_file="$3"
  "$PY" - "$pair_file" "$other_file" "$libero10_file" <<'PYEOF'
import sys

pair_file, other_file, libero10_file = sys.argv[1:]
other, libero10 = [], []
for line in open(pair_file, encoding="utf-8"):
    pair = line.strip()
    if not pair:
        continue
    if pair.startswith("libero_10:"):
        libero10.append(pair)
    else:
        other.append(pair)
with open(other_file, "w", encoding="utf-8") as f:
    for pair in other:
        f.write(pair + "\n")
with open(libero10_file, "w", encoding="utf-8") as f:
    for pair in libero10:
        f.write(pair + "\n")
print(len(other), len(libero10))
PYEOF
}

aggregate_standard() {
  local out="$1"
  "$PY" - "$out" "$out/missing_standard_pairs.txt" <<'PYEOF'
import glob
import json
import os
import sys

out, missing_path = sys.argv[1], sys.argv[2]
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
expected = [(suite, task_id) for suite in suites for task_id in range(10)]
seen = {}
for path in sorted(glob.glob(os.path.join(out, "results_*.json")) + glob.glob(os.path.join(out, "results_partial_*.json"))):
    if os.path.basename(path).startswith("summary_"):
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    for row in data.get("by_task", []):
        key = (row["suite"], int(row["task_id"]))
        if key not in seen and int(row.get("trials", 0)) >= 50:
            seen[key] = row
missing = [f"{suite}:{task_id}" for suite, task_id in expected if (suite, task_id) not in seen]
by_suite = {}
tot_s = tot_t = 0
for suite, task_id in expected:
    row = seen.get((suite, task_id))
    if not row:
        continue
    s, t = int(row["successes"]), int(row["trials"])
    item = by_suite.setdefault(suite, {"successes": 0, "trials": 0})
    item["successes"] += s
    item["trials"] += t
    tot_s += s
    tot_t += t
for item in by_suite.values():
    item["rate"] = item["successes"] / max(item["trials"], 1)
summary = {
    "complete": not missing,
    "tasks": len(seen),
    "missing_tasks": len(missing),
    "by_suite": by_suite,
    "overall": {"successes": tot_s, "trials": tot_t, "rate": tot_s / max(tot_t, 1)},
}
json.dump(summary, open(os.path.join(out, "summary_aggregate.json"), "w", encoding="utf-8"), indent=2)
with open(missing_path, "w", encoding="utf-8") as f:
    for pair in missing:
        f.write(pair + "\n")
lines = []
for suite in suites:
    item = by_suite.get(suite, {"successes": 0, "trials": 0, "rate": 0.0})
    lines.append("[standard] %-16s %4d/%4d = %.2f%%" % (suite, item["successes"], item["trials"], 100 * item["rate"]))
lines.append("[standard] OVERALL          %4d/%4d = %.2f%% complete=%s missing=%d" % (tot_s, tot_t, 100 * summary["overall"]["rate"], not missing, len(missing)))
text = "\n".join(lines)
print(text)
open(os.path.join(out, "summary_aggregate.txt"), "w", encoding="utf-8").write(text + "\n")
PYEOF
}

aggregate_plus() {
  local out="$1"
  "$PY" - "$out" "$LIBERO_PLUS_ROOT" "$out/missing_plus_pairs.txt" <<'PYEOF'
import glob
import json
import os
import sys

out, root, missing_path = sys.argv[1], sys.argv[2], sys.argv[3]
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
cls = json.load(open(root + "/libero/libero/benchmark/task_classification.json", encoding="utf-8"))
expected = []
cat_map = {}
for suite in suites:
    for task_id, entry in enumerate(cls[suite]):
        expected.append((suite, task_id))
        cat_map[(suite, task_id)] = entry.get("category")
seen = {}
for path in sorted(glob.glob(os.path.join(out, "results_*.json")) + glob.glob(os.path.join(out, "results_partial_*.json"))):
    if os.path.basename(path).startswith("summary_"):
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    for row in data.get("by_task", []):
        key = (row["suite"], int(row["task_id"]))
        if key not in seen:
            seen[key] = row
missing = [f"{suite}:{task_id}" for suite, task_id in expected if (suite, task_id) not in seen]
by_suite, by_cat = {}, {}
tot_s = tot_t = 0
for suite, task_id in expected:
    row = seen.get((suite, task_id))
    if not row:
        continue
    s, t = int(row["successes"]), int(row["trials"])
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
summary = {
    "complete": not missing,
    "tasks": len(seen),
    "expected_tasks": len(expected),
    "missing_tasks": len(missing),
    "by_category": by_cat,
    "by_suite": by_suite,
    "overall": {"successes": tot_s, "trials": tot_t, "rate": tot_s / max(tot_t, 1)},
}
json.dump(summary, open(os.path.join(out, "summary_aggregate.json"), "w", encoding="utf-8"), indent=2)
with open(missing_path, "w", encoding="utf-8") as f:
    for pair in missing:
        f.write(pair + "\n")
lines = ["tasks aggregated: %d/%d complete=%s missing=%d" % (len(seen), len(expected), not missing, len(missing)), "", "== per dimension =="]
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
print(text)
open(os.path.join(out, "summary_aggregate.txt"), "w", encoding="utf-8").write(text + "\n")
PYEOF
}

eval_standard_step() {
  local step="$1"
  local out="$REPO/evaluate_results/${RUN_ID}_step$(printf "%06d" "$step")_libero_standard50_robust_${RUN_TS}"
  mkdir -p "$out"
  echo "$out" > /data/users/junjie/current_eval_standard_out.txt
  echo "$out/eval_main.log" > /data/users/junjie/current_eval_standard_log.txt
  echo "[standard] step=$step out=$out"
  write_standard_pairs "$out"
  local pair_file="$out/all_standard_pairs.txt"
  local attempt
  for attempt in $(seq 0 "$STD_RETRIES"); do
    run_shards standard "$out" "$step" "$pair_file" "$STANDARD_NPROC" "$attempt"
    aggregate_standard "$out" | tee -a "$out/eval_main.log"
    local missing
    missing="$(pair_count "$out/missing_standard_pairs.txt")"
    [ "$missing" -eq 0 ] && return 0
    echo "[standard] step=$step missing=$missing after attempt=$attempt"
    pair_file="$out/missing_standard_pairs.txt"
  done
  echo "[standard] step=$step incomplete after retries"
  return 1
}

eval_plus_step() {
  local step="$1"
  local out="$REPO/evaluate_results/${RUN_ID}_step$(printf "%06d" "$step")_libero_plus_full10030_robust_${RUN_TS}"
  mkdir -p "$out"
  echo "$out" > /data/users/junjie/current_eval_plus_out.txt
  echo "$out/eval_main.log" > /data/users/junjie/current_eval_plus_log.txt
  echo "[plus] step=$step out=$out"
  write_plus_pairs "$out"
  local pair_file="$out/all_plus_pairs.txt"
  local attempt
  for attempt in $(seq 0 "$PLUS_RETRIES"); do
    if [ -n "$PLUS_LIBERO10_GPU_LIST" ]; then
      local plus_other_pairs="$out/plus_pairs_non_libero10_a${attempt}.txt"
      local plus_libero10_pairs="$out/plus_pairs_libero10_a${attempt}.txt"
      read -r other_count libero10_count < <(split_plus_pairs_by_suite "$pair_file" "$plus_other_pairs" "$plus_libero10_pairs")
      echo "[plus] split attempt=$attempt non_libero10=$other_count libero10=$libero10_count"
      if [ "$other_count" -gt 0 ]; then
        run_shards plus "$out" "$step" "$plus_other_pairs" "$PLUS_NPROC" "$attempt"
      fi
      if [ "$libero10_count" -gt 0 ]; then
        run_shards plus_libero10 "$out" "$step" "$plus_libero10_pairs" "$PLUS_LIBERO10_NPROC" "$attempt"
      fi
    else
      run_shards plus "$out" "$step" "$pair_file" "$PLUS_NPROC" "$attempt"
    fi
    aggregate_plus "$out" | tee -a "$out/eval_main.log"
    local missing
    missing="$(pair_count "$out/missing_plus_pairs.txt")"
    [ "$missing" -eq 0 ] && return 0
    echo "[plus] step=$step missing=$missing after attempt=$attempt"
    pair_file="$out/missing_plus_pairs.txt"
  done
  echo "[plus] step=$step incomplete after retries"
  return 1
}

steps="$(make_steps)"
echo "[robust-eval] queue steps: $steps"
for step in $steps; do
  ckpt="$RUN_DIR/checkpoints/weights/step_$(printf "%06d" "$step").pt"
  if [ ! -f "$ckpt" ]; then
    echo "[robust-eval] skip missing ckpt: $ckpt"
    continue
  fi
  echo "[robust-eval] ===== step=$step start $(date) ====="
  if [ "$SKIP_STANDARD" = "1" ]; then
    echo "[robust-eval] SKIP_STANDARD=1; skipping standard step=$step"
    std_rc=0
  else
    eval_standard_step "$step"
    std_rc="$?"
  fi
  if [ "$SKIP_PLUS" = "1" ]; then
    echo "[robust-eval] SKIP_PLUS=1; skipping plus step=$step"
    plus_rc=0
  else
    eval_plus_step "$step"
    plus_rc="$?"
  fi
  echo "[robust-eval] ===== step=$step done standard_rc=$std_rc plus_rc=$plus_rc $(date) ====="
done
echo "[robust-eval] all queued steps done $(date)"
