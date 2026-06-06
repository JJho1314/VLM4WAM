#!/usr/bin/env python3
"""Evaluate InstructSAM referring-segmentation quality (IoU / hit-rate) against GT.

Consumes the SAME json that build_instructsam_sft_data.py emits (image + phrase +
GT RLE), so frames are pre-extracted PNGs (no video reader needed in the
instructsam env). For each record it runs the model on the image with the
phrase query, picks the best mask (combine_mode='best' = argmax cls_score, like
the Cosmos bridge), and compares to the GT mask.

Metrics (referring-seg standard):
  gIoU  : mean of per-image IoU            (primary)
  cIoU  : sum(intersection)/sum(union)     (cumulative)
  Pr@K  : fraction of images with IoU >= K  (hit rate)
  found : fraction where the model returned a non-empty mask

Usage (run in the instructsam env):
  EVAL:
    python tools/eval_instructsam_iou.py \
        --data-json data/eval_holdout/sft_data.json --data-root data/eval_holdout \
        --model-path <base_or_finetuned_model> --out results_base.json
  COMPARE two result files (before/after):
    python tools/eval_instructsam_iou.py --compare results_base.json results_ft.json

Tip: build an eval json from a HELD-OUT split (val is fine for *eval*):
    /opt/miniconda3/bin/python tools/build_instructsam_sft_data.py \
        --dataset-dir <val_one_per_scene_dir> --out-dir data/eval_holdout --frames-per-episode 3
"""
import argparse, json, os, sys
import numpy as np


def cmd_compare(a_path, b_path):
    a = json.load(open(a_path)); b = json.load(open(b_path))
    A, B = a["aggregate"], b["aggregate"]
    print(f"{'metric':<12}{'before':>10}{'after':>10}{'delta':>10}")
    print("-" * 42)
    for k in ["n", "found_rate", "gIoU", "cIoU", "mIoU_found", "Pr@0.25", "Pr@0.5", "Pr@0.7"]:
        va, vb = A.get(k), B.get(k)
        if va is None or vb is None:
            continue
        d = vb - va if k != "n" else vb - va
        print(f"{k:<12}{va:>10.4f}{vb:>10.4f}{d:>+10.4f}")
    print(f"\nbefore: {a_path}\nafter : {b_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE.json", "AFTER.json"),
                    help="print delta table between two --out result files and exit")
    ap.add_argument("--data-json")
    ap.add_argument("--data-root", default=None, help="root for relative image paths (default: dir of --data-json)")
    ap.add_argument("--model-path", default="/data/LFT-W02_data/junjie/weights/CircleRadon/InstructSAM-2B")
    ap.add_argument("--repo", default="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/InstructSAM")
    ap.add_argument("--out", default="instructsam_eval_results.json")
    ap.add_argument("--max-records", type=int, default=0, help="0 = all")
    ap.add_argument("--query-template", default="Please segment '{phrase}' in the image.")
    ap.add_argument("--sam-letterbox", type=int, default=0, help="0=official stretch (default), 1=letterbox")
    args = ap.parse_args()

    if args.compare:
        cmd_compare(*args.compare); return

    assert args.data_json, "--data-json required for eval"
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from pycocotools import mask as maskUtils
    sys.path.insert(0, args.repo)
    from instructsam.models import load_pretrained_model
    from instructsam import mm_infer_segmentation

    data_root = args.data_root or os.path.dirname(os.path.abspath(args.data_json))
    _txt = open(args.data_json).read().strip()
    records = json.loads(_txt) if _txt[:1] == "[" else [json.loads(l) for l in _txt.splitlines() if l.strip()]
    if args.max_records:
        records = records[: args.max_records]

    print(f"[load] {args.model_path}", flush=True)
    tokenizer, model, processor = load_pretrained_model(args.model_path, None, attn_implementation="sdpa")
    model.to(torch.bfloat16).eval()
    print(f"[load] done; {len(records)} records", flush=True)

    def decode_gt(ann_list, h, w):
        gt = np.zeros((h, w), np.uint8)
        for a in ann_list:
            for rle_str in a.get("ann", []):
                rle = json.loads(rle_str)
                m = maskUtils.decode(maskUtils.frPyObjects(rle, h, w)) if isinstance(rle["counts"], list) else maskUtils.decode(rle)
                gt |= (np.asarray(m) > 0).astype(np.uint8)
        return gt

    def best_pred(pred_masks, cls_score, h, w):
        if pred_masks is None:
            return np.zeros((h, w), np.uint8)
        m = pred_masks.detach().float().cpu().reshape(-1, *pred_masks.shape[-2:])
        s = cls_score.detach().float().cpu().reshape(-1)
        best = m[int(s.argmax()):int(s.argmax()) + 1].unsqueeze(0)
        if best.shape[-2:] != (h, w):
            best = F.interpolate(best, size=(h, w), mode="nearest")
        return (best[0, 0] > 0).numpy().astype(np.uint8)

    per, inter_sum, union_sum = [], 0, 0
    for i, r in enumerate(records):
        h, w = int(r["height"]), int(r["width"])
        phrase = r["annotation"][0]["text"]
        gt = decode_gt(r["annotation"], h, w)
        img_path = os.path.join(data_root, r["image"])
        query = args.query_template.format(phrase=phrase)
        conv = [{"role": "user", "content": [{"type": "image", "image": img_path}, {"type": "text", "text": query}]}]
        with torch.inference_mode():
            _out, pred_masks, cls_score = mm_infer_segmentation(
                img_path, processor, conv, model, tokenizer, sam_letterbox=bool(args.sam_letterbox))
        pred = best_pred(pred_masks, cls_score, h, w)
        inter = int(np.logical_and(pred, gt).sum()); union = int(np.logical_or(pred, gt).sum())
        iou = inter / union if union > 0 else (1.0 if pred.sum() == gt.sum() == 0 else 0.0)
        inter_sum += inter; union_sum += union
        per.append({"image": r["image"], "phrase": phrase, "iou": iou,
                    "pred_px": int(pred.sum()), "gt_px": int(gt.sum()), "found": bool(pred.sum() > 0)})
        if (i + 1) % 10 == 0 or i + 1 == len(records):
            print(f"[{i+1}/{len(records)}] running gIoU={np.mean([p['iou'] for p in per]):.4f}", flush=True)

    ious = np.array([p["iou"] for p in per])
    found = np.array([p["found"] for p in per])
    agg = {
        "n": len(per),
        "found_rate": float(found.mean()),
        "gIoU": float(ious.mean()),
        "cIoU": float(inter_sum / union_sum) if union_sum > 0 else 0.0,
        "mIoU_found": float(ious[found].mean()) if found.any() else 0.0,
        "median_IoU": float(np.median(ious)),
        "Pr@0.25": float((ious >= 0.25).mean()),
        "Pr@0.5": float((ious >= 0.5).mean()),
        "Pr@0.7": float((ious >= 0.7).mean()),
    }
    json.dump({"model_path": args.model_path, "data_json": args.data_json,
               "sam_letterbox": bool(args.sam_letterbox), "aggregate": agg, "per_record": per},
              open(args.out, "w"), indent=2)
    print("\n=== aggregate ===")
    for k, v in agg.items():
        print(f"  {k:<12} {v}")
    print(f"\n[done] -> {args.out}")


if __name__ == "__main__":
    main()
