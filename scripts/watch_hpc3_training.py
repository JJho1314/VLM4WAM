#!/usr/bin/env python3
"""Lightweight Slurm/log watchdog for the HPC3 target-feature training run."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


ERROR_RE = re.compile(r"\b(traceback|exception|failed|killed|oom|cuda out of memory|nan)\b", re.IGNORECASE)
PROCESSED_RE = re.compile(r"\[rank\s+(\d+)\]\s+processed=(\d+)\s+skipped=(\d+)\s+errors=(\d+)")
ITER_RE = re.compile(r"\b(?:iter|iteration)\D{0,8}(\d+)", re.IGNORECASE)
LOSS_RE = re.compile(r"\b(loss|target_attention_loss|target_feature_contrastive_loss)\b", re.IGNORECASE)


def run(command: list[str]) -> str:
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.stdout.strip()


def tail(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return ""
    return run(["tail", f"-{lines}", str(path)])


def count_pt(path: Path) -> int:
    if not path.exists():
        return -1
    return sum(1 for item in path.iterdir() if item.suffix == ".pt")


def summarize_precompute(log_text: str) -> tuple[int, int, int, int]:
    latest_by_rank: dict[int, tuple[int, int, int]] = {}
    for match in PROCESSED_RE.finditer(log_text):
        rank = int(match.group(1))
        latest_by_rank[rank] = (int(match.group(2)), int(match.group(3)), int(match.group(4)))
    processed = sum(value[0] for value in latest_by_rank.values())
    skipped = sum(value[1] for value in latest_by_rank.values())
    errors = sum(value[2] for value in latest_by_rank.values())
    return len(latest_by_rank), processed, skipped, errors


def find_bad_lines(text: str, limit: int = 20) -> list[str]:
    bad = [line for line in text.splitlines() if ERROR_RE.search(line)]
    return bad[-limit:]


def summarize_training(text: str) -> tuple[str, str]:
    last_iter = "unknown"
    for match in ITER_RE.finditer(text):
        last_iter = match.group(1)
    loss_lines = [line for line in text.splitlines() if LOSS_RE.search(line)]
    return last_iter, loss_lines[-1] if loss_lines else ""


def write_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-job", default="332166")
    parser.add_argument("--train-job", default="332167")
    parser.add_argument("--log-dir", type=Path, default=Path("/data/user/jli545/workspace/VLM4VLA/slurm_logs"))
    parser.add_argument("--train-feature-dir", type=Path, required=True)
    parser.add_argument("--val-feature-dir", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=72355)
    parser.add_argument("--expected-val", type=int, default=977)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--max-checks", type=int, default=288)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pre_out = args.log_dir / f"gtmask-features-v3-{args.pre_job}.out"
    pre_err = args.log_dir / f"gtmask-features-v3-{args.pre_job}.err"
    train_out = args.log_dir / f"gtmask-feature-target-branch-{args.train_job}.out"
    train_err = args.log_dir / f"gtmask-feature-target-branch-{args.train_job}.err"

    stable_training_checks = 0
    for check_idx in range(1, args.max_checks + 1):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        queue = run(["squeue", "-j", f"{args.pre_job},{args.train_job}", "-o", "%.18i %.9P %.45j %.8u %.2t %.12M %.6D %R"])
        acct = run(["sacct", "-j", f"{args.pre_job},{args.train_job}", "--format=JobID,JobName%32,State,ExitCode,Elapsed,NodeList", "-P"])

        pre_text = "\n".join([tail(pre_out, 300), tail(pre_err, 120)])
        train_text = "\n".join([tail(train_out, 300), tail(train_err, 120)])
        ranks, processed, skipped, pre_errors = summarize_precompute(pre_text)
        train_count = count_pt(args.train_feature_dir)
        val_count = count_pt(args.val_feature_dir)
        bad_lines = find_bad_lines(pre_text + "\n" + train_text)
        last_iter, last_loss_line = summarize_training(train_text)

        train_running = re.search(rf"^\s*{re.escape(args.train_job)}\s+.*\sR\s", queue, re.MULTILINE) is not None
        train_completed = f"{args.train_job}|cosmos-gtfeat-branch|COMPLETED|" in acct
        train_failed = re.search(rf"{re.escape(args.train_job)}(?:\.\w+)?\|.*\|(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY)", acct) is not None

        if train_running and last_iter != "unknown" and not bad_lines:
            stable_training_checks += 1
        else:
            stable_training_checks = 0

        status = (
            f"[{now}] check={check_idx} features=train:{train_count}/{args.expected_train} "
            f"val:{val_count}/{args.expected_val} pre_tail_ranks={ranks} "
            f"pre_tail_processed={processed} pre_tail_skipped={skipped} pre_tail_errors={pre_errors} "
            f"train_running={train_running} train_iter={last_iter} stable_training_checks={stable_training_checks}"
        )
        write_line(args.output, status)
        if last_loss_line:
            write_line(args.output, f"[{now}] loss_line={last_loss_line}")
        if bad_lines:
            write_line(args.output, f"[{now}] ALERT suspicious log lines:")
            for line in bad_lines:
                write_line(args.output, f"  {line}")
        if train_failed:
            write_line(args.output, f"[{now}] ALERT training job failed according to sacct")
            return 2
        if train_completed:
            write_line(args.output, f"[{now}] training job completed")
            return 0
        if stable_training_checks >= 3:
            write_line(args.output, f"[{now}] training appears stable for {stable_training_checks} consecutive checks")
            return 0

        time.sleep(max(30, args.interval_sec))
    write_line(args.output, f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] watchdog reached max checks")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
