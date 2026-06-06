"""Prepare zero-shot inputs for ALL 120 RoboInter demo videos.

For each video:
- Extract a frame at t=2s (or middle if shorter) — by t=2s the arm has typically
  entered the scene for action clips, more robust than the very first frame.
- Use the LMDB `instruction_add` annotation, wrapped in an arm-subject prompt
  template matching the v2 manifest convention.
- Skip videos already prepared (idempotent).
- Also copy GT video to the comparison dir.
"""
import json
import shutil
import subprocess
from pathlib import Path

import lmdb
import pickle

VIDEO_DIR = Path(
    "/data/LFT-W02_data/junjie/data/InternRobotics/RoboInter-Data/Annotation_demo_larger/videos"
)
LMDB_DIR = Path(
    "/data/LFT-W02_data/junjie/data/InternRobotics/RoboInter-Data/Annotation_demo_larger/demo_annotations"
)

OUT_DIR = Path("/data/LFT-W02_data/junjie/data/robointer_test_inputs")
GT_DIR = Path("/data/LFT-W02_data/junjie/cosmos-predict2.5/outputs/robointer_zero_shot_gt")
OUT_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)


def video_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def extract_frame(video: Path, ts: float, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(ts), "-i", str(video), "-frames:v", "1", str(dst)],
        check=True,
    )


def is_droid(name: str) -> bool:
    return not name.startswith("RH20T_")


def make_prompt(name: str, instr: str | None) -> str:
    if not instr:
        instr = "perform a manipulation task"
    instr = instr.strip().rstrip(".")
    if is_droid(name):
        # DROID is mostly Franka on a counter / table
        return f"A Franka robotic arm with a parallel-jaw gripper {instr}."
    # RH20T has 2 arms in many cells; use neutral two-arm subject
    return f"Two robotic arms with parallel-jaw grippers; one of them {instr}."


def main():
    env = lmdb.open(str(LMDB_DIR), readonly=True, lock=False, readahead=False)
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"Total videos: {len(videos)}")
    n_new = 0
    for v in videos:
        name = v.stem
        png = OUT_DIR / f"{name}.png"
        cfg = OUT_DIR / f"{name}.json"
        if png.exists() and cfg.exists():
            continue  # already prepared

        # Choose timestamp for the conditioning frame
        try:
            dur = video_duration(v)
        except Exception:
            dur = 3.0
        ts = 2.0 if dur > 2.5 else dur / 2.0

        try:
            extract_frame(v, ts, png)
        except subprocess.CalledProcessError:
            print(f"[skip extract failed] {name}")
            continue

        # Pull instruction from LMDB
        with env.begin() as txn:
            raw = txn.get(name.encode())
        instr = None
        if raw is not None:
            data = pickle.loads(raw)
            f0 = data[sorted(data.keys())[0]]
            instr = f0.get("instruction_add") or f0.get("substask")
        prompt = make_prompt(name, instr)

        (OUT_DIR / f"{name}.txt").write_text(prompt + "\n")
        cfg.write_text(
            json.dumps(
                {
                    "inference_type": "image2world",
                    "name": name,
                    "input_path": f"{name}.png",
                    "prompt_path": f"{name}.txt",
                },
                indent=2,
            )
            + "\n"
        )

        # GT for side-by-side
        gt_dst = GT_DIR / f"{name}_gt.mp4"
        if not gt_dst.exists():
            shutil.copyfile(v, gt_dst)
        n_new += 1

    print(f"New samples prepared: {n_new}")
    print(f"Total JSONs: {len(list(OUT_DIR.glob('*.json')))}")


if __name__ == "__main__":
    main()
