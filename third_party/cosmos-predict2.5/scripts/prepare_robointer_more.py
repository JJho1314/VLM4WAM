"""Prepare 4 more zero-shot samples (where arm is visible)."""
import json
import shutil
import subprocess
from pathlib import Path

VIDEO_DIR = Path(
    "/data/LFT-W02_data/junjie/data/InternRobotics/RoboInter-Data/Annotation_demo_larger/videos"
)
SCOUT_DIR = Path("/data/LFT-W02_data/junjie/Wan2.2/data/zero_shot_robointer/scout")

OUT_DIR = Path("/data/LFT-W02_data/junjie/data/robointer_test_inputs")
GT_DIR = Path("/data/LFT-W02_data/junjie/cosmos-predict2.5/outputs/robointer_zero_shot_gt")

# (name, frame_strategy, instruction_orig, prompt)
SAMPLES = [
    (
        "26076_exterior_image_2_left",
        "t2",
        "put the glass lid on the black pot",
        "A Franka robotic arm with a parallel-jaw gripper picks up the glass lid from the table and places it on top of the black pot.",
    ),
    (
        "37401_exterior_image_1_left",
        "t2",
        "folding the towel on the table",
        "A Franka robotic arm with a parallel-jaw gripper folds the towel on the counter, grasping a corner and bringing it across.",
    ),
    (
        "71070_exterior_image_1_left",
        "scout",
        "put the scissors inside the mug, then take it out and place it on the table",
        "A Franka robotic arm with a parallel-jaw gripper picks up the scissors from the table and places them inside the green mug.",
    ),
    (
        "RH20T_cfg5_task_0011_user_0010_scene_0010_cfg_0005_104122063678",
        "scout",
        "water the plant",
        "Two robotic arms with parallel-jaw grippers; one arm picks up the green watering can from the table to water the plant.",
    ),
]


def extract_at(video: Path, ts: float, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(ts), "-i", str(video), "-frames:v", "1", str(dst)],
        check=True,
    )


def main():
    for name, strat, _instr, prompt in SAMPLES:
        png = OUT_DIR / f"{name}.png"
        if strat == "scout":
            shutil.copyfile(SCOUT_DIR / f"{name}.png", png)
        elif strat == "t2":
            extract_at(VIDEO_DIR / f"{name}.mp4", 2.0, png)

        (OUT_DIR / f"{name}.txt").write_text(prompt + "\n")
        (OUT_DIR / f"{name}.json").write_text(
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

        # GT video for side-by-side
        shutil.copyfile(VIDEO_DIR / f"{name}.mp4", GT_DIR / f"{name}_gt.mp4")
        print(f"prepared: {name} ({strat}) | {prompt}")

    print(f"\nTotal new: {len(SAMPLES)} samples")


if __name__ == "__main__":
    main()
