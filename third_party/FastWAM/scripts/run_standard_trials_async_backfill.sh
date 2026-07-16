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
TRIALS_PER_JOB="${TRIALS_PER_JOB:-3}"
STANDARD_GPU_LIST="${STANDARD_GPU_LIST:-3,4,5,6,7}"
STANDARD_SET_MUJOCO_EGL_DEVICE_ID="${STANDARD_SET_MUJOCO_EGL_DEVICE_ID:-1}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-0}"
POLL_SECONDS="${POLL_SECONDS:-5}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

source "$TRAIN_ENV"
cd "$REPO" || exit 2
mkdir -p "$REPO/evaluate_results/auto_eval_logs"
ASYNC_LOG="${ASYNC_LOG:-$REPO/evaluate_results/auto_eval_logs/standard_async_$(date +%Y%m%d_%H%M%S).log}"
echo "$ASYNC_LOG" > /data/users/junjie/current_eval_standard_log.txt
exec > >(tee -a "$ASYNC_LOG") 2>&1

for d in "$VENV"/lib/python3.10/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export MAGICK_HOME="${MAGICK_HOME:-/data/users/junjie/im_env}"
export LD_LIBRARY_PATH="/data/users/junjie/im_env/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset LIBERO_CONFIG_PATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export REPO PY RUN_ID RUN_DIR START_STEP MIN_STEP STEP_INTERVAL NIS NUM_TRIALS TRIALS_PER_JOB ASYNC_LOG
export STANDARD_GPU_LIST STANDARD_SET_MUJOCO_EGL_DEVICE_ID JOB_TIMEOUT_SECONDS POLL_SECONDS
export ACTION_HIDDEN_DIM ACTION_FFN_DIM ACTION_ATTENTION_HEAD_DIM

"$PY" - <<'PYEOF'
import glob
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict

repo = os.environ.get("REPO", "/data/users/junjie/FastWAM_cosmos")
py = os.environ.get("PY", "/data/users/junjie/cosmos-predict2.5-fw/.venv/bin/python")
run_id = os.environ["RUN_ID"]
run_dir = os.environ["RUN_DIR"]
start_step = int(os.environ.get("START_STEP", "14600"))
min_step = int(os.environ.get("MIN_STEP", "11200"))
step_interval = int(os.environ.get("STEP_INTERVAL", "200"))
nis = int(os.environ.get("NIS", "10"))
num_trials = int(os.environ.get("NUM_TRIALS", "50"))
trials_per_job = max(1, int(os.environ.get("TRIALS_PER_JOB", "3")))
gpus = [x for x in os.environ.get("STANDARD_GPU_LIST", "3,4,5,6,7").split(",") if x]
set_egl = os.environ.get("STANDARD_SET_MUJOCO_EGL_DEVICE_ID", "1") == "1"
timeout_s = int(os.environ.get("JOB_TIMEOUT_SECONDS", "0") or 0)
poll_s = max(1, int(os.environ.get("POLL_SECONDS", "5")))
action_hidden_dim = os.environ.get("ACTION_HIDDEN_DIM", "1024")
action_ffn_dim = os.environ.get("ACTION_FFN_DIM", "4096")
action_attention_head_dim = os.environ.get("ACTION_ATTENTION_HEAD_DIM", "128")

suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
tasks = [(suite, task_id) for suite in suites for task_id in range(10)]

if not gpus:
    raise SystemExit("empty STANDARD_GPU_LIST")

def result_files(out):
    files = glob.glob(os.path.join(out, "results_partial_*.json"))
    files += glob.glob(os.path.join(out, "results_*.json"))
    return [p for p in files if not os.path.basename(p).startswith("summary_")]

def parse_standard(out):
    expected = {(suite, task_id, trial) for suite, task_id in tasks for trial in range(num_trials)}
    seen = {}
    bad = []
    for path in sorted(result_files(out)):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
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
    return expected, seen, bad

def write_summary(out):
    expected, seen, bad = parse_standard(out)
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
    with open(os.path.join(out, "summary_aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out, "missing_standard_trials.live.txt"), "w", encoding="utf-8") as f:
        for suite, task_id, trial in sorted(expected - set(seen), key=lambda x: (x[2], x[0], x[1])):
            f.write(f"{suite}:{task_id}:{trial}\n")
    lines = [
        f"trial units aggregated: {len(seen)}/{len(expected)} complete={summary['complete']} bad_files={len(bad)}",
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
    print(text, flush=True)
    return summary

def complete_existing(step):
    pattern = os.path.join(
        repo,
        "evaluate_results",
        f"{run_id}_step{step:06d}_libero_standard50_*",
        "summary_aggregate.json",
    )
    for path in sorted(glob.glob(pattern), reverse=True):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        overall = data.get("overall", {})
        trials = int(overall.get("trials", 0) or 0)
        trial_units = int(data.get("trial_units", 0) or 0)
        tasks_done = int(data.get("tasks", 0) or 0)
        if (tasks_done == 40 and trials == 2000) or (trial_units == 2000 and trials == 2000):
            return os.path.dirname(path)
    return ""

def choose_job(expected, seen, active_keys):
    seen_keys = set(seen)
    blocked = seen_keys | active_keys
    for suite, task_id in tasks:
        trial = 0
        while trial < num_trials:
            while trial < num_trials and (suite, task_id, trial) in blocked:
                trial += 1
            if trial >= num_trials:
                break
            start = trial
            keys = []
            while trial < num_trials and len(keys) < trials_per_job and (suite, task_id, trial) not in blocked:
                keys.append((suite, task_id, trial))
                trial += 1
            if keys:
                return suite, task_id, start, len(keys), keys
    return None

def launch_job(out, step, gpu, job, launch_idx):
    suite, task_id, trial_start, trials, keys = job
    tag = f"standard_async_step{step}_n{launch_idx}_{int(time.time())}_trial{trial_start}_{suite}_{task_id}"
    log = os.path.join(out, f"proc_{tag}_gpu{gpu}.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if set_egl:
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    else:
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    env.pop("LIBERO_CONFIG_PATH", None)
    cmd = [
        py,
        "experiments/libero/cosmos_eval_libero.py",
        "--pairs", f"{suite}:{task_id}",
        "--tag", tag,
        "--num_trials", str(trials),
        "--trial_start", str(trial_start),
        "--num_inference_steps", str(nis),
        "--coupling", "mot",
        "--run_dir", run_dir,
        "--step", str(step),
        "--out_dir", out,
        "--action_hidden_dim", str(action_hidden_dim),
        "--action_ffn_dim", str(action_ffn_dim),
        "--action_attention_head_dim", str(action_attention_head_dim),
        "--no-save_videos",
    ]
    f = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(
        f"[standard-async] launch gpu={gpu} pid={proc.pid} step={step} "
        f"pair={suite}:{task_id} trial={trial_start} trials={trials} log={log}",
        flush=True,
    )
    return {"proc": proc, "log_file": f, "log": log, "keys": set(keys), "start": time.time(), "gpu": gpu, "job": job}

def run_step(step, out):
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(repo, "evaluate_results", "auto_eval_logs"), exist_ok=True)
    print(f"[standard-async] step={step} out={out} gpus={','.join(gpus)} trials_per_job={trials_per_job}", flush=True)
    print(out, file=open("/data/users/junjie/current_eval_standard_out.txt", "w", encoding="utf-8"))
    active = {}
    launch_idx = 0
    last_report = 0.0
    failed = 0
    timed_out = 0
    while True:
        expected, seen, bad = parse_standard(out)
        if len(seen) >= len(expected) and not bad:
            for info in active.values():
                info["proc"].terminate()
            write_summary(out)
            return True

        now = time.time()
        for gpu, info in list(active.items()):
            proc = info["proc"]
            rc = proc.poll()
            if rc is None and timeout_s > 0 and now - info["start"] > timeout_s:
                proc.terminate()
                time.sleep(2)
                if proc.poll() is None:
                    proc.kill()
                rc = proc.wait()
                timed_out += 1
                print(f"[standard-async] timeout gpu={gpu} pid={proc.pid} rc={rc} log={info['log']}", flush=True)
            if rc is not None:
                info["log_file"].close()
                if rc != 0:
                    failed += 1
                    print(f"[standard-async] finished-failed gpu={gpu} pid={proc.pid} rc={rc} log={info['log']}", flush=True)
                else:
                    print(f"[standard-async] finished-ok gpu={gpu} pid={proc.pid} log={info['log']}", flush=True)
                active.pop(gpu, None)

        expected, seen, bad = parse_standard(out)
        active_keys = set()
        for info in active.values():
            active_keys |= info["keys"]
        for gpu in gpus:
            if gpu in active:
                continue
            job = choose_job(expected, seen, active_keys)
            if job is None:
                continue
            active[gpu] = launch_job(out, step, gpu, job, launch_idx)
            active_keys |= active[gpu]["keys"]
            launch_idx += 1

        if now - last_report > 60:
            succ = sum(seen.values())
            print(
                f"[standard-async] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"step={step} units={len(seen)}/{len(expected)} succ={succ} "
                f"active={len(active)} failed={failed} timed_out={timed_out} bad={len(bad)}",
                flush=True,
            )
            write_summary(out)
            last_report = now
        time.sleep(poll_s)

ts = time.strftime("%Y%m%d_%H%M%S")
step = start_step
while step >= min_step:
    existing = complete_existing(step)
    if existing:
        print(f"[standard-async] step={step} already complete: {existing}", flush=True)
        step -= step_interval
        continue
    current_out = os.environ.get("OUT", "").strip()
    if current_out and f"step{step:06d}" in current_out:
        out = current_out
    elif step == start_step:
        pointer = "/data/users/junjie/current_eval_standard_out.txt"
        pointed = open(pointer, encoding="utf-8").read().strip() if os.path.exists(pointer) else ""
        out = pointed if pointed and f"step{step:06d}" in pointed else ""
        if not out:
            out = os.path.join(repo, "evaluate_results", f"{run_id}_step{step:06d}_libero_standard50_async_{ts}")
    else:
        out = os.path.join(repo, "evaluate_results", f"{run_id}_step{step:06d}_libero_standard50_async_{ts}")
    ok = run_step(step, out)
    if not ok:
        raise SystemExit(3)
    step -= step_interval

print("[standard-async] complete", flush=True)
PYEOF
