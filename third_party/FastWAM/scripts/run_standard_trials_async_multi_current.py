#!/usr/bin/env python3
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict


REPO = os.environ.get("REPO", "/data/users/junjie/FastWAM_cosmos")
PY = os.environ.get("PY", "/data/users/junjie/cosmos-predict2.5-fw/.venv/bin/python")
OUT = os.environ["OUT"]
STEP = int(os.environ["STEP"])
RUN_DIR = os.environ["RUN_DIR"]
NIS = int(os.environ.get("NIS", "10"))
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", "50"))
TRIALS_PER_JOB = max(1, int(os.environ.get("TRIALS_PER_JOB", "1")))
POLL_SECONDS = max(1, int(os.environ.get("POLL_SECONDS", "5")))
REPORT_SECONDS = max(10, int(os.environ.get("REPORT_SECONDS", "60")))
TARGET_PER_GPU = int(os.environ.get("TARGET_PER_GPU", "10"))
GPU_LIST = [x for x in os.environ.get("STANDARD_GPU_LIST", "0,1,2,3,4,5,6,7").split(",") if x]
SET_EGL = os.environ.get("STANDARD_SET_MUJOCO_EGL_DEVICE_ID", "1") == "1"
TAG_BASE = os.environ.get("TAG_BASE", f"standard_async_multi_{time.strftime('%Y%m%d_%H%M%S')}")
ACTION_HIDDEN_DIM = os.environ.get("ACTION_HIDDEN_DIM", "1024")
ACTION_FFN_DIM = os.environ.get("ACTION_FFN_DIM", "4096")
ACTION_ATTENTION_HEAD_DIM = os.environ.get("ACTION_ATTENTION_HEAD_DIM", "128")

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TASKS = [(suite, task_id) for suite in SUITES for task_id in range(10)]


def result_files(out):
    files = glob.glob(os.path.join(out, "results_partial_*.json"))
    files += glob.glob(os.path.join(out, "results_*.json"))
    return [p for p in files if not os.path.basename(p).startswith("summary_")]


def parse_standard(out):
    expected = {(suite, task_id, trial) for suite, task_id in TASKS for trial in range(NUM_TRIALS)}
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
        success = int(seen[(suite, task_id, trial)])
        by_task[(suite, task_id)]["successes"] += success
        by_task[(suite, task_id)]["trials"] += 1
        by_suite[suite]["successes"] += success
        by_suite[suite]["trials"] += 1
    for group in (by_task, by_suite):
        for item in group.values():
            item["rate"] = item["successes"] / max(item["trials"], 1)
    total_success = sum(seen.values())
    total_trials = len(seen)
    summary = {
        "complete": len(seen) == len(expected) and not bad,
        "tasks": len(by_task),
        "trial_units": len(seen),
        "expected_trial_units": len(expected),
        "missing_trial_units": len(expected) - len(seen),
        "bad_files": bad,
        "by_task": {f"{suite}:{task_id}": item for (suite, task_id), item in sorted(by_task.items())},
        "by_suite": dict(sorted(by_suite.items())),
        "overall": {
            "successes": total_success,
            "trials": total_trials,
            "rate": total_success / max(total_trials, 1),
        },
    }
    with open(os.path.join(out, "summary_aggregate.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out, "missing_standard_trials.live.txt"), "w", encoding="utf-8") as f:
        for suite, task_id, trial in sorted(expected - set(seen), key=lambda x: (x[2], x[0], x[1])):
            f.write(f"{suite}:{task_id}:{trial}\n")
    lines = [
        "trial units aggregated: %d/%d complete=%s bad_files=%d"
        % (len(seen), len(expected), summary["complete"], len(bad)),
        "",
        "== per suite ==",
    ]
    for suite in SUITES:
        item = by_suite.get(suite, {"successes": 0, "trials": 0, "rate": 0.0})
        lines.append("  %-16s %4d/%4d = %.2f%%" % (suite, item["successes"], item["trials"], 100 * item["rate"]))
    lines.append("")
    lines.append("OVERALL %d/%d = %.2f%%" % (total_success, total_trials, 100 * summary["overall"]["rate"]))
    text = "\n".join(lines)
    with open(os.path.join(out, "summary_aggregate.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return summary, text


def arg_value(cmd, name):
    toks = cmd.split()
    for idx, tok in enumerate(toks):
        if tok == name and idx + 1 < len(toks):
            return toks[idx + 1]
        if tok.startswith(name + "="):
            return tok.split("=", 1)[1]
    return ""


def parse_pairs(pairs):
    out = []
    for item in pairs.split(","):
        if ":" not in item:
            continue
        suite, task = item.split(":", 1)
        try:
            out.append((suite, int(task)))
        except ValueError:
            continue
    return out


def gpu_uuid_map():
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    mapping = {}
    for line in text.strip().splitlines():
        if "," not in line:
            continue
        idx, uuid = [x.strip() for x in line.split(",", 1)]
        mapping[uuid] = idx
    return mapping


def scan_live_eval(out):
    uuid_to_idx = gpu_uuid_map()
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        text = ""
    pid_to_gpu = {}
    for line in text.strip().splitlines():
        if "," not in line:
            continue
        pid, uuid = [x.strip() for x in line.split(",", 1)]
        pid_to_gpu[int(pid)] = uuid_to_idx.get(uuid, "")

    keys = set()
    counts = Counter()
    pids = set()
    for pid, gpu in pid_to_gpu.items():
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
        except Exception:
            continue
        if "cosmos_eval_libero.py" not in cmd or out not in cmd:
            continue
        pairs = parse_pairs(arg_value(cmd, "--pairs"))
        if not pairs:
            continue
        try:
            trial_start = int(arg_value(cmd, "--trial_start") or 0)
            trials = int(arg_value(cmd, "--num_trials") or 1)
        except ValueError:
            continue
        for suite, task_id in pairs:
            for trial in range(trial_start, trial_start + trials):
                keys.add((suite, task_id, trial))
        if gpu:
            counts[gpu] += 1
        pids.add(pid)
    return keys, counts, pids


def choose_job(expected, seen, blocked):
    seen_keys = set(seen)
    unavailable = seen_keys | blocked
    for suite, task_id in TASKS:
        trial = 0
        while trial < NUM_TRIALS:
            while trial < NUM_TRIALS and (suite, task_id, trial) in unavailable:
                trial += 1
            if trial >= NUM_TRIALS:
                break
            start = trial
            keys = []
            while trial < NUM_TRIALS and len(keys) < TRIALS_PER_JOB and (suite, task_id, trial) not in unavailable:
                keys.append((suite, task_id, trial))
                trial += 1
            if keys:
                return suite, task_id, start, len(keys), keys
    return None


def launch(out, gpu, job, launch_idx):
    suite, task_id, trial_start, trials, keys = job
    tag = f"{TAG_BASE}_n{launch_idx}_{int(time.time())}_{suite}_{task_id}_trial{trial_start}"
    log = os.path.join(out, f"proc_{tag}_gpu{gpu}.log")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if SET_EGL:
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    else:
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    env.pop("LIBERO_CONFIG_PATH", None)
    cmd = [
        PY,
        "experiments/libero/cosmos_eval_libero.py",
        "--pairs",
        f"{suite}:{task_id}",
        "--tag",
        tag,
        "--num_trials",
        str(trials),
        "--trial_start",
        str(trial_start),
        "--num_inference_steps",
        str(NIS),
        "--coupling",
        "mot",
        "--run_dir",
        RUN_DIR,
        "--step",
        str(STEP),
        "--out_dir",
        out,
        "--action_hidden_dim",
        str(ACTION_HIDDEN_DIM),
        "--action_ffn_dim",
        str(ACTION_FFN_DIM),
        "--action_attention_head_dim",
        str(ACTION_ATTENTION_HEAD_DIM),
        "--no-save_videos",
    ]
    f = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(
        f"[standard-async-multi] launch gpu={gpu} pid={proc.pid} pair={suite}:{task_id} "
        f"trial={trial_start} trials={trials} log={log}",
        flush=True,
    )
    return {
        "proc": proc,
        "log_file": f,
        "log": log,
        "gpu": str(gpu),
        "keys": set(keys),
        "start": time.time(),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    print(
        f"[standard-async-multi] step={STEP} out={OUT} gpus={','.join(GPU_LIST)} "
        f"target_per_gpu={TARGET_PER_GPU} trials_per_job={TRIALS_PER_JOB}",
        flush=True,
    )
    active = {}
    launch_idx = 0
    failed = 0
    last_report = 0.0
    while True:
        now = time.time()
        for pid, info in list(active.items()):
            rc = info["proc"].poll()
            if rc is None:
                continue
            info["log_file"].close()
            active.pop(pid, None)
            if rc != 0:
                failed += 1
                print(f"[standard-async-multi] finished-failed gpu={info['gpu']} pid={pid} rc={rc} log={info['log']}", flush=True)
            else:
                print(f"[standard-async-multi] finished-ok gpu={info['gpu']} pid={pid} log={info['log']}", flush=True)

        expected, seen, bad = parse_standard(OUT)
        live_keys, live_counts, live_pids = scan_live_eval(OUT)
        active_keys = set(live_keys)
        counts = Counter(live_counts)
        for pid, info in active.items():
            if pid not in live_pids:
                active_keys |= info["keys"]
                counts[info["gpu"]] += 1

        if len(seen) >= len(expected) and not bad and not active:
            summary, text = write_summary(OUT)
            print(text, flush=True)
            print("[standard-async-multi] complete", flush=True)
            return 0

        for gpu in GPU_LIST:
            while counts[str(gpu)] < TARGET_PER_GPU:
                job = choose_job(expected, seen, active_keys)
                if job is None:
                    break
                info = launch(OUT, str(gpu), job, launch_idx)
                active[info["proc"].pid] = info
                active_keys |= info["keys"]
                counts[str(gpu)] += 1
                launch_idx += 1

        if now - last_report >= REPORT_SECONDS:
            summary, text = write_summary(OUT)
            print(
                f"[standard-async-multi] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"units={summary['trial_units']}/{summary['expected_trial_units']} "
                f"missing={summary['missing_trial_units']} active={sum(counts.values())} "
                f"counts={dict(sorted(counts.items()))} failed={failed} bad={len(bad)}",
                flush=True,
            )
            print(text, flush=True)
            last_report = now
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
