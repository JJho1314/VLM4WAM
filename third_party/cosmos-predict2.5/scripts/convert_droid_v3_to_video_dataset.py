"""Convert LeRobot DROID v3.0 (multiple episodes per mp4) into the format Cosmos
VideoDataset expects (one episode per mp4 + a matching prompt file).

For each episode in meta/episodes/chunk-XXX/file_YYY.parquet we ffmpeg-cut the
view's source mp4 to the episode's [from_timestamp, to_timestamp] range using
stream copy (no re-encoding, ~50 ms per cut). Episodes shorter than --min-frames
are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


PROMPT_TEMPLATE = "A Franka robotic arm with a parallel-jaw gripper {task}."


def find_ffmpeg() -> str:
    candidates = [
        os.environ.get("FFMPEG"),
        shutil.which("ffmpeg"),
        "/data/apps/ffmpeg/7.0.2/ffmpeg",
        "/data/apps/ffmpeg/6.0.1/ffmpeg",
        "/data/apps/ffmpeg/5.1.1/ffmpeg",
        "/data/apps/ffmpeg/4.4.1/ffmpeg",
        "/data/apps/ffmpeg/5.1.7/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("ffmpeg not found; set FFMPEG=/path/to/ffmpeg")


def normalize_task(task: str) -> str:
    # `task` field is "<inst_a>|<inst_b>|<inst_c>"; pick the first as canonical
    first = task.split("|")[0].strip().rstrip(".").strip()
    if not first:
        return "performs a manipulation task"
    return first[0].lower() + first[1:]


def cut_episode(args):
    ffmpeg, src_mp4, dst_mp4, from_ts, to_ts, height, width, fps, preset, crf, threads, timeout_sec = args
    if dst_mp4.exists() and dst_mp4.stat().st_size > 0:
        return "skip"
    tmp_mp4 = dst_mp4.with_suffix(".tmp.mp4")
    tmp_mp4.unlink(missing_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{from_ts:.3f}", "-to", f"{to_ts:.3f}", "-i", str(src_mp4)]
    if height is not None and width is not None:
        cmd += [
            "-vf",
            f"fps={fps},scale={width}:{height}:flags=bicubic,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-threads",
            str(threads),
            "-an",
            "-movflags",
            "+faststart",
        ]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(tmp_mp4))
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout_sec)
        tmp_mp4.replace(dst_mp4)
        return "ok"
    except subprocess.CalledProcessError as e:
        tmp_mp4.unlink(missing_ok=True)
        return f"err: {e.stderr.decode()[:200]}"
    except Exception as e:
        tmp_mp4.unlink(missing_ok=True)
        return f"err: {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/user/jhe724/workspace/data/droid_success")
    ap.add_argument("--out", required=True, help="Output dataset dir")
    ap.add_argument("--view", default="observation.images.left_external")
    ap.add_argument("--episode-list", default=None, help="Text file with one episode_index per line to include")
    ap.add_argument("--exclude-episode-list", default=None, help="Text file with episode_index values to exclude")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--min-frames", type=int, default=33)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--timeout-sec", type=int, default=600)
    args = ap.parse_args()
    if (args.height is None) != (args.width is None):
        ap.error("--height and --width must be set together")

    src = Path(args.src)
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "metas").mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    print(f"using ffmpeg: {ffmpeg}")

    info = json.loads((src / "meta" / "info.json").read_text())
    fps = info["fps"]

    # Collect all episode rows from parquet shards
    eps_dir = src / "meta" / "episodes"
    rows = []
    for chunk_dir in sorted(eps_dir.iterdir()):
        for f in sorted(chunk_dir.glob("*.parquet")):
            rows.append(pd.read_parquet(f))
    df = pd.concat(rows, ignore_index=True)
    print(f"loaded {len(df)} episode rows from meta")

    if args.episode_list:
        include_ids = {int(line.strip()) for line in Path(args.episode_list).read_text().splitlines() if line.strip()}
        df = df[df["episode_index"].astype(int).isin(include_ids)]
        print(f"filtered to {len(df)} rows from --episode-list")
    if args.exclude_episode_list:
        exclude_ids = {int(line.strip()) for line in Path(args.exclude_episode_list).read_text().splitlines() if line.strip()}
        df = df[~df["episode_index"].astype(int).isin(exclude_ids)]
        print(f"filtered to {len(df)} rows after --exclude-episode-list")

    chunk_col = f"videos/{args.view}/chunk_index"
    file_col = f"videos/{args.view}/file_index"
    from_col = f"videos/{args.view}/from_timestamp"
    to_col = f"videos/{args.view}/to_timestamp"

    # Filter & build cut tasks
    cut_jobs = []
    csv_rows = []
    n_short = n_missing = 0
    for _, ep in df.iterrows():
        if args.max_episodes is not None and len(cut_jobs) >= args.max_episodes:
            break
        if int(ep["length"]) < args.min_frames:
            n_short += 1
            continue
        chunk = int(ep[chunk_col])
        file_idx = int(ep[file_col])
        src_mp4 = (
            src / "videos" / args.view / f"chunk-{chunk:03d}" / f"file_{file_idx:03d}.mp4"
        )
        if not src_mp4.exists():
            n_missing += 1
            continue
        ep_idx = int(ep["episode_index"])
        name = f"episode_{ep_idx:06d}"
        dst_mp4 = out / "videos" / f"{name}.mp4"
        cut_jobs.append(
            (
                ffmpeg,
                src_mp4,
                dst_mp4,
                float(ep[from_col]),
                float(ep[to_col]),
                args.height,
                args.width,
                args.fps,
                args.preset,
                args.crf,
                args.threads,
                args.timeout_sec,
            )
        )

        # Write prompt
        prompt = PROMPT_TEMPLATE.format(task=normalize_task(ep["task"]))
        (out / "metas" / f"{name}.txt").write_text(prompt + "\n")
        csv_rows.append((name, int(ep["length"]), ep["task"].split("|")[0]))

    print(f"to cut: {len(cut_jobs)}  short(<{args.min_frames}): {n_short}  missing: {n_missing}")

    # Parallel cut
    n_ok = n_skip = n_err = 0
    err_samples = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(cut_episode, j) for j in cut_jobs]
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            if r == "ok":
                n_ok += 1
            elif r == "skip":
                n_skip += 1
            else:
                n_err += 1
                if len(err_samples) < 5:
                    err_samples.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  progress: {i+1}/{len(futures)} ok={n_ok} skip={n_skip} err={n_err}")

    # metadata.csv
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "length", "task_orig"])
        w.writerows(csv_rows)

    print(f"\nDone: ok={n_ok} skipped={n_skip} err={n_err}")
    if err_samples:
        print("first errors:")
        for e in err_samples:
            print("  ", e)
    print(f"output: {out}")


if __name__ == "__main__":
    main()
