#!/usr/bin/env python3
"""Build InstructSAM referring-segmentation SFT data from a one-per-scene-style
dataset (videos/*.mp4 + masks/*.npz + one_per_scene_manifest.json + frame_ranges.json).

For each scene it samples frames, pairs each frame with its per-frame target mask,
and writes records in the exact schema consumed by instructsam.training.dataset
(`_convert_conversation`):

  {
    "image": "images/<name>_fNNNNNN.png",     # relative to --out-dir (=data_root)
    "height": H, "width": W,
    "annotation": [
      {"type": "phrase", "text": "<phrase>", "ann_type": "mask",
       "ann": ["<json-string of uncompressed COCO RLE>"]}
    ]
  }

Outputs under --out-dir:
  images/...                  extracted frames
  sft_data.json               JSON array of records
  data_list.txt               ann_path file (one line: "sft_data.json 1")

Run with base miniconda python (has decord + pycocotools + PIL):
  /opt/miniconda3/bin/python tools/build_instructsam_sft_data.py \
      --dataset-dir <one_per_scene_dir> --out-dir data/robot_sft --frames-per-episode 6

WARNING: do NOT point --dataset-dir at a validation/holdout split you evaluate on.
"""
import argparse, json, os, sys
import gc
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

try:
    from decord import VideoReader, cpu
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False
    import imageio.v3 as iio


def load_packed_masks(npz_path):
    """Return [N, H, W] uint8 binary masks from {masks_packed, shape}."""
    z = np.load(npz_path, allow_pickle=True)
    if "masks_packed" in z.files and "shape" in z.files:
        shape = tuple(int(x) for x in z["shape"].tolist())
        flat = int(np.prod(shape[1:]))
        arr = np.unpackbits(z["masks_packed"], axis=1)[:, :flat].reshape(shape)
    elif "masks" in z.files:
        arr = z["masks"]
    else:
        arr = z[z.files[0]]
    return (np.asarray(arr) > 0).astype(np.uint8)


def mask_to_uncompressed_rle(mask):
    """Binary HxW mask -> COCO uncompressed RLE dict {size, counts:[int,...]}.

    COCO RLE is column-major (Fortran) run lengths, the first run counting 0s.
    annToMask handles this via the `isinstance(counts, list)` -> frPyObjects path.
    """
    h, w = int(mask.shape[0]), int(mask.shape[1])
    flat = np.asfortranarray(mask.astype(np.uint8)).ravel(order="F")
    if flat.size == 0:
        return {"size": [h, w], "counts": [0]}
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).astype(np.int64).tolist()
    if flat[0] != 0:
        runs = [0] + runs
    return {"size": [h, w], "counts": [int(c) for c in runs]}


def resize_mask_nearest(mask, out_h, out_w):
    if (mask.shape[0], mask.shape[1]) == (out_h, out_w):
        return mask
    pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize((out_w, out_h), Image.NEAREST)
    return (np.asarray(pil) > 0).astype(np.uint8)


def read_frames(video_path, indices):
    if _HAS_DECORD:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        n = len(vr)
        idx = [min(i, n - 1) for i in indices]
        batch = vr.get_batch(idx).asnumpy()
        del vr  # decord leaks if readers aren't dropped promptly
        return batch, n
    frames = iio.imread(video_path, plugin="pyav")
    n = len(frames)
    return np.stack([frames[min(i, n - 1)] for i in indices]), n


def sample_indices(ranges, length, k):
    pool = []
    for a, b in ranges:
        pool.extend(range(max(0, a), min(length, b + 1)))
    pool = sorted(set(pool)) or list(range(length))
    if len(pool) <= k:
        return pool
    sel = np.linspace(0, len(pool) - 1, k).round().astype(int)
    return [pool[i] for i in sorted(set(sel.tolist()))]


def phrase_for_scene(scene):
    ph = (scene.get("rewrite_object_phrase") or "").strip()
    if ph:
        return ph
    cap = scene.get("target_caption", "")
    if "[TGT]" in cap:  # take a few words after the [TGT] marker
        return cap.split("[TGT]", 1)[1].strip().rstrip(".").split(",")[0].strip()
    return (scene.get("task_orig", "the target object") or "the target object").strip()


def load_scenes(d):
    """Return (scenes, frame_ranges). Supports two layouts:
    1) one-per-scene: *manifest*.json with a 'scenes' list (name, rewrite_object_phrase).
    2) train-dir: tgt_caption_rewrite_report_*.csv (path->object_phrase) + frame_ranges.json
       + masks/ + videos/ (the droid_*_train layout).
    """
    import csv, glob
    fr_path = os.path.join(d, "frame_ranges.json")
    frame_ranges = json.load(open(fr_path)) if os.path.exists(fr_path) else {}

    manifest = next((os.path.join(d, f) for f in os.listdir(d)
                     if "manifest" in f and f.endswith(".json")), None)
    scenes = []
    if manifest:
        try:
            scenes = json.load(open(manifest)).get("scenes", [])
        except Exception:
            scenes = []
    if scenes:
        return scenes, frame_ranges

    # train-dir mode: phrases from the rewrite report; episodes from masks/ that have a video.
    report = sorted(glob.glob(os.path.join(d, "tgt_caption_rewrite_report_*.csv")))
    assert report, f"no manifest scenes and no tgt_caption_rewrite_report_*.csv in {d}"
    name2phrase = {}
    with open(report[-1]) as f:
        for row in csv.DictReader(f):
            stem = os.path.splitext(os.path.basename(row.get("path", "")))[0]
            ph = (row.get("object_phrase") or "").strip()
            if stem and ph:
                name2phrase[stem] = ph
    excl = set()
    exf = os.path.join(d, "exclude_no_tgt_manifest.json")
    if os.path.exists(exf):
        excl = set(json.load(open(exf)).get("stems", []))
    # Derive episodes from the rewrite report (in-memory; no per-file stat over the
    # 88k-file masks/ dir). The main loop lazily skips any missing video/mask.
    scenes = [{"name": n, "rewrite_object_phrase": p}
              for n, p in sorted(name2phrase.items()) if n not in excl]
    return scenes, frame_ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="one-per-scene-style dir (videos/, masks/, *manifest*.json, frame_ranges.json)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames-per-episode", type=int, default=6)
    ap.add_argument("--min-mask-pixels", type=int, default=80)
    ap.add_argument("--camera", default=None, help="optional substring filter, e.g. left_external")
    ap.add_argument("--limit", type=int, default=0, help="cap number of episodes (0=all), for a quick test")
    ap.add_argument("--max-episodes", type=int, default=0, help="alias of --limit (0=all)")
    ap.add_argument("--start-episode", type=int, default=0, help="offset into the (filtered) episode list; for chunked builds")
    ap.add_argument("--append", action="store_true", help="append to the jsonl instead of overwriting (chunked builds)")
    ap.add_argument("--jsonl-name", default="sft_data.jsonl", help="output jsonl filename (parallel workers use distinct names sharing one images/ dir)")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth episode (diversity without scanning all)")
    args = ap.parse_args()

    d = args.dataset_dir
    scenes, frame_ranges = load_scenes(d)

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    if args.camera:
        scenes = [s for s in scenes if args.camera in s.get("name", "")]
    if args.stride > 1:
        scenes = scenes[:: args.stride]
    cap = args.max_episodes or args.limit
    lo = args.start_episode
    hi = lo + cap if cap else len(scenes)
    scenes = scenes[lo:hi]
    print(f"[scenes] {len(scenes)} episodes (slice [{lo}:{hi}]) to process")

    # Stream records to JSONL (flat memory). Chunked builds (--append) restart a
    # fresh process per slice so decord's per-reader memory is released between chunks.
    kept_scenes, n_records, verified = 0, 0, (args.append and args.start_episode > 0)
    jsonl_path = os.path.join(args.out_dir, args.jsonl_name)
    out_f = open(jsonl_path, "a" if args.append else "w")

    for si, s in enumerate(scenes):
        name = s["name"]
        video = os.path.join(d, "videos", f"{name}.mp4")
        npz = os.path.join(d, "masks", f"{name}.npz")
        if not (os.path.exists(video) and os.path.exists(npz)):
            continue
        phrase = phrase_for_scene(s)
        try:                                                  # one bad/corrupt video must not kill the chunk
            masks = load_packed_masks(npz)                    # [N,Hm,Wm]
            length = int(s.get("length", masks.shape[0]))
            ranges = frame_ranges.get(name, [[0, length - 1]])
            idxs = sample_indices(ranges, min(length, masks.shape[0]), args.frames_per_episode)
            frames, vlen = read_frames(video, idxs)           # [k,H,W,3]
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:80]}")
            continue
        H, W = frames.shape[1], frames.shape[2]
        n_added = 0
        for j, fi in enumerate(idxs):
            if fi >= masks.shape[0]:
                continue
            m = resize_mask_nearest(masks[fi], H, W)
            if int(m.sum()) < args.min_mask_pixels:
                continue
            rle = mask_to_uncompressed_rle(m)
            if not verified:  # sanity: decode back via the same path training uses
                dec = maskUtils.decode(maskUtils.merge(maskUtils.frPyObjects(rle, H, W))) \
                    if False else _decode_like_anntomask(rle, H, W)
                inter = np.logical_and(dec > 0, m > 0).sum(); union = np.logical_or(dec > 0, m > 0).sum()
                iou = inter / max(1, union)
                assert iou > 0.999, f"RLE round-trip IoU={iou:.4f} (encoding bug)"
                print(f"[verify] RLE round-trip IoU={iou:.4f} OK")
                verified = True
            rel = f"images/{name}_f{fi:06d}.png"
            Image.fromarray(frames[j]).save(os.path.join(args.out_dir, rel))
            out_f.write(json.dumps({
                "type": "refseg",   # top-level dataset type; anything != "instseg" -> _convert_conversation
                "image": rel, "height": H, "width": W,
                "annotation": [{"type": "phrase", "text": phrase, "ann_type": "mask",
                                "ann": [json.dumps(rle)]}],
            }) + "\n")
            n_added += 1
            n_records += 1
        if n_added:
            kept_scenes += 1
        del masks, frames
        if si % 25 == 0:            # periodic gc/flush: bounds decord leak without per-scene overhead
            gc.collect()
            out_f.flush()
            print(f"[progress] processed={si+1} kept={kept_scenes} records={n_records}")

    out_f.close()
    if args.jsonl_name == "sft_data.jsonl":   # parallel workers skip; orchestrator writes data_list after concat
        with open(os.path.join(args.out_dir, "data_list.txt"), "w") as f:
            f.write("sft_data.jsonl 1\n")
    print(f"\n[done] scenes_kept={kept_scenes} records={n_records}")
    print(f"       data={jsonl_path}")
    print(f"       ann_path={os.path.join(args.out_dir,'data_list.txt')}")


def _decode_like_anntomask(mask_ann, h, w):
    """Mirror instructsam.training.mm_utils.annToMask for an uncompressed RLE dict."""
    if isinstance(mask_ann["counts"], list):
        rle = maskUtils.frPyObjects(mask_ann, h, w)
    else:
        rle = mask_ann
    return maskUtils.decode(rle)


if __name__ == "__main__":
    main()
