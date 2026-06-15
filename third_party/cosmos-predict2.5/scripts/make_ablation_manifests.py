#!/usr/bin/env python3
"""Build manifests for the text-free multisource target-query ablation.

Emits:
  - stage1 manifest.json: [{image, query, out}]  (feature extraction, InstructSAM env)
  - stage2 samples.jsonl:  one InferenceArguments per line  (generation, cosmos venv)

Each holdout scene is generated twice from the SAME conditioning video: once
conditioned on the REAL target's fused feature, once on a DISTRACTOR object's
feature. If generation follows the guided target, the InstructSAM features steer
the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (episode, real_phrase, distractor_phrase) chosen for clear, distinct objects.
SCENES = [
    ("episode_006447_left_external", "the red object", "the green building block"),
    ("episode_006448_left_external", "the white rope", "the black board"),
    ("episode_006413_right_external", "the cubes", "the bowl"),
    ("episode_006455_left_external", "the rope", "the black object"),
    ("episode_006425_right_external", "the blocks", "the black bowl"),
    ("episode_006409_right_external", "the toy blocks", "the bowl"),
]

QUERY_TMPL = "Please segment '{phrase}' in the image."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-dir", required=True, help="Holdout dataset dir (has videos/ and metas/).")
    ap.add_argument("--work-dir", required=True, help="Output dir for features/, manifests, videos.")
    ap.add_argument("--num-frames", type=int, default=49)
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Use only the first N scenes (0=all).")
    args = ap.parse_args()

    val = Path(args.val_dir)
    work = Path(args.work_dir)
    feat_dir = work / "features"
    inputs_dir = work / "inputs"
    feat_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # First frames are extracted lazily (decord available in the cosmos venv).
    import decord
    from PIL import Image

    decord.bridge.set_bridge("native")

    scenes = SCENES[: args.limit] if args.limit > 0 else SCENES
    manifest = []      # stage1
    samples = []       # stage2
    summary = []       # per-scene record of all inputs
    for ep, real_phrase, distractor_phrase in scenes:
        video = val / "videos" / f"{ep}.mp4"
        if not video.exists():
            raise FileNotFoundError(f"Missing holdout video: {video}")
        meta = val / "metas" / f"{ep}.txt"
        caption = meta.read_text().strip() if meta.exists() else ""
        prompt = caption.replace("[TGT] ", "").replace("[TGT]", "").strip()

        # Conditioning must be the FIRST frame (image2world). Feeding the full GT
        # mp4 as video2world conditions on the LAST frames (video-extension
        # semantics) — the task-completed state — which is wrong for this eval.
        first_frame = inputs_dir / f"{ep}__first_frame.png"
        if not first_frame.exists():
            vr = decord.VideoReader(str(video), num_threads=2)
            Image.fromarray(vr.get_batch([0]).asnumpy()[0]).save(first_frame)
        (inputs_dir / f"{ep}__caption.txt").write_text(caption + "\n")

        scene_rec = {"episode": ep, "caption": caption, "gt_video": str(video),
                     "first_frame": str(first_frame), "conditions": {}}
        for cond, phrase in (("real", real_phrase), ("distractor", distractor_phrase)):
            out_feat = feat_dir / f"{ep}__{cond}.pt"
            query = QUERY_TMPL.format(phrase=phrase)
            manifest.append({"image": str(video), "query": query, "out": str(out_feat)})
            samples.append({
                "name": f"{ep}__{cond}",
                "inference_type": "image2world",
                "input_path": str(first_frame),
                "prompt": prompt,                       # ignored (text-free model), kept for logs
                "target_feature_path": str(out_feat),
                "num_output_frames": args.num_frames,
                "num_steps": args.num_steps,
                "guidance": args.guidance,
                "seed": args.seed,
            })
            scene_rec["conditions"][cond] = {
                "phrase": phrase, "query": query, "feature": str(out_feat),
                "mask_png": str(out_feat.with_suffix("")) + "_mask.png",
            }
        summary.append(scene_rec)

    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (work / "samples.jsonl").open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    (work / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"scenes={len(scenes)}  stage1_pairs={len(manifest)}  stage2_samples={len(samples)}")
    print(f"manifest -> {work/'manifest.json'}")
    print(f"samples  -> {work/'samples.jsonl'}")
    print(f"summary  -> {work/'summary.json'}  (captions/first frames in {inputs_dir})")
    for s in scenes:
        print(f"  {s[0]}: real={s[1]!r}  distractor={s[2]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
