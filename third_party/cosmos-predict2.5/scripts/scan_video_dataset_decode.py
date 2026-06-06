#!/usr/bin/env python3
"""Scan a Cosmos VideoDataset directory for videos that hang or fail decoding."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import signal
import time
import traceback
from pathlib import Path


def check_video(path: str, num_frames: int, random_windows: int, queue: mp.Queue) -> None:
    try:
        import numpy as np
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0), num_threads=2)
        total = len(vr)
        if total < num_frames:
            queue.put({"status": "short", "frames": total, "path": path})
            return

        max_start = total - num_frames
        starts = {0, max_start // 2, max_start}
        rng = random.Random(hash(path) & 0xFFFFFFFF)
        for _ in range(random_windows):
            starts.add(rng.randint(0, max_start))

        decoded = 0
        for start in sorted(starts):
            frame_ids = np.arange(start, start + num_frames).tolist()
            arr = vr.get_batch(frame_ids).asnumpy()
            if arr.shape[0] != num_frames:
                queue.put(
                    {
                        "status": "bad_shape",
                        "frames": total,
                        "start": start,
                        "shape": tuple(arr.shape),
                        "path": path,
                    }
                )
                return
            decoded += 1

        try:
            fps = vr.get_avg_fps()
        except Exception:
            fps = None
        queue.put({"status": "ok", "frames": total, "fps": fps, "windows": decoded, "path": path})
    except Exception as exc:
        queue.put(
            {
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(limit=8),
                "path": path,
            }
        )


def terminate_process(proc: mp.Process) -> None:
    if proc.pid is not None:
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    proc.join(timeout=2)
    if proc.is_alive() and proc.pid is not None:
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--random-windows", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    video_dir = Path(args.dataset_dir) / "videos"
    paths = sorted(str(path) for path in video_dir.glob("*.mp4"))
    if args.limit is not None:
        paths = paths[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pending = iter(paths)
    running: list[tuple[mp.Process, mp.Queue, str, float]] = []
    done = 0
    bad = 0
    started = time.time()

    def start_next() -> bool:
        try:
            path = next(pending)
        except StopIteration:
            return False
        queue: mp.Queue = mp.Queue(maxsize=1)
        proc = mp.Process(target=check_video, args=(path, args.num_frames, args.random_windows, queue))
        proc.start()
        running.append((proc, queue, path, time.time()))
        return True

    for _ in range(min(args.workers, len(paths))):
        start_next()

    with out_path.open("w") as out_file:
        while running:
            now = time.time()
            next_running: list[tuple[mp.Process, mp.Queue, str, float]] = []
            for proc, queue, path, start_time in running:
                result = None
                if not proc.is_alive():
                    proc.join(timeout=0)
                    if not queue.empty():
                        result = queue.get()
                    else:
                        result = {"status": "no_result", "exitcode": proc.exitcode, "path": path}
                elif now - start_time > args.timeout_sec:
                    terminate_process(proc)
                    result = {"status": "timeout", "timeout_sec": args.timeout_sec, "path": path}
                else:
                    next_running.append((proc, queue, path, start_time))
                    continue

                done += 1
                if result["status"] != "ok":
                    bad += 1
                    print("BAD", json.dumps(result, ensure_ascii=False), flush=True)
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()
                start_next()

            running = next_running
            if done and done % 250 == 0:
                elapsed = time.time() - started
                print(f"checked={done}/{len(paths)} bad={bad} elapsed={elapsed:.1f}s", flush=True)
            time.sleep(0.05)

    elapsed = time.time() - started
    print(f"DONE checked={done} bad={bad} elapsed={elapsed:.1f}s out={out_path}", flush=True)
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
