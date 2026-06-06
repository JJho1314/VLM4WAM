"""Prepare Cosmos zero-shot inputs aligned with the Wan2.2 manifest_v2.

Uses the v2 manifest where:
  * the source first frame clearly shows the robot arm, and
  * the prompt's subject is the robot arm itself
    (e.g. "A Franka robotic arm with a parallel-jaw gripper picks up...").

Reuses the same frames Wan2.2 was given so the two models see identical inputs.
Also collects GT source videos into a sibling folder for side-by-side review.
"""
import json
import shutil
from pathlib import Path

WAN_ROOT = Path("/data/LFT-W02_data/junjie/Wan2.2/data/zero_shot_robointer")
MANIFEST = WAN_ROOT / "manifest_v2.json"

OUT_DIR = Path("/data/LFT-W02_data/junjie/data/robointer_test_inputs")
GT_DIR = Path("/data/LFT-W02_data/junjie/cosmos-predict2.5/outputs/robointer_zero_shot_gt")

OUT_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    samples = json.loads(MANIFEST.read_text())
    keep = {s["name"] for s in samples}

    for s in samples:
        name = s["name"]
        prompt = s["prompt"]  # arm-subject prompt from v2

        # Reuse the v2 first-frame PNG for identical input
        shutil.copyfile(s["frame"], OUT_DIR / f"{name}.png")
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

        # Copy the GT video for side-by-side viewing.
        shutil.copyfile(s["source_video"], GT_DIR / f"{name}_gt.mp4")
        print(f"prepared: {name}\n  prompt: {prompt}")

    # Drop stale inputs from earlier sample selections.
    for f in OUT_DIR.iterdir():
        if f.stem not in keep:
            f.unlink()
            print(f"removed stale: {f.name}")

    print(f"\nTotal: {len(samples)} samples")
    print(f"  inputs: {OUT_DIR}")
    print(f"  GT videos: {GT_DIR}")


if __name__ == "__main__":
    main()
