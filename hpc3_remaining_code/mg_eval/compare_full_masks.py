"""Compare InstructSAM first-frame masks vs GT over the FULL droid set; emit
per-stem IoU records, per-split stats, and filtered correct-episode lists."""

import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

FULL = "/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864"
DS = "/data/user/jhe724/workspace/datasets"
OUT_DIR = os.path.join(FULL, "instructsam_mask_eval")
os.makedirs(OUT_DIR, exist_ok=True)


def stem_set(d, sub="videos"):
    return {os.path.basename(p).rsplit(".", 1)[0] for p in glob.glob(f"{DS}/{d}/{sub}/*.mp4")}


def one(stem):
    try:
        z = np.load(f"{FULL}/masks/{stem}.npz")
        _, H, W = [int(v) for v in z["shape"]]
        gt = np.unpackbits(z["masks_packed"][0])[: H * W].reshape(H, W) > 0
        gt_area = int(gt.sum())
        pred_path = f"{FULL}/target_features_ft/{stem}_mask.png"
        if not os.path.exists(pred_path):
            return {"stem": stem, "status": "no_pred"}
        pm = np.array(Image.open(pred_path).resize((W, H), Image.NEAREST)) > 127
        pred_area = int(pm.sum())
        if gt_area == 0:
            return {"stem": stem, "status": "gt_empty", "pred_area": pred_area}
        union = int((gt | pm).sum())
        iou = float((gt & pm).sum()) / union if union else 0.0
        score = None
        try:
            d = torch.load(f"{FULL}/target_features_ft/{stem}.pt", map_location="cpu", weights_only=False)
            score = float(d.get("score") or 0.0)
        except Exception:
            pass
        return {"stem": stem, "status": "ok", "iou": round(iou, 4), "score": score,
                "gt_area": gt_area, "pred_area": pred_area}
    except Exception as e:
        return {"stem": stem, "status": f"error:{type(e).__name__}"}


def main():
    stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{FULL}/videos/*.mp4"))
    print(f"total stems: {len(stems)}", flush=True)
    with Pool(32) as pool:
        rows = pool.map(one, stems, chunksize=64)

    with open(f"{OUT_DIR}/per_stem_iou.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    splits = {
        "cap200_train": stem_set("droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_scene_cap200_tasktarget"),
        "train_strict_v3": stem_set("droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3"),
        "val_strict_v3": stem_set("droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3"),
        "val_full": stem_set("droid_success_v21_target_aware_left_right_480x864_val"),
    }

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"compared={len(ok)} no_pred={sum(r['status']=='no_pred' for r in rows)} "
          f"gt_empty={sum(r['status']=='gt_empty' for r in rows)} "
          f"errors={sum(r['status'].startswith('error') for r in rows)}", flush=True)

    def report(name, sel):
        if not sel:
            print(f"{name}: empty")
            return
        iou = np.array([r["iou"] for r in sel])
        line = (f"{name}: n={len(sel)} | mean={iou.mean():.3f} median={np.median(iou):.3f} | "
                f">=0.5: {(iou>=0.5).mean()*100:.1f}% | >=0.3: {(iou>=0.3).mean()*100:.1f}% | "
                f"<0.05: {(iou<0.05).mean()*100:.1f}%")
        print(line, flush=True)
        return line

    lines = [report("ALL", ok)]
    for name, ss in splits.items():
        lines.append(report(name, [r for r in ok if r["stem"] in ss]))
    rest = splits["train_strict_v3"] - splits["cap200_train"]
    lines.append(report("train_strict_v3_minus_cap200", [r for r in ok if r["stem"] in rest]))

    for thr, tag in ((0.5, "iou50"), (0.3, "iou30")):
        keep = sorted(r["stem"] for r in ok if r["iou"] >= thr)
        with open(f"{OUT_DIR}/mask_correct_stems_{tag}.txt", "w") as f:
            f.write("\n".join(keep) + "\n")
        print(f"saved mask_correct_stems_{tag}.txt: {len(keep)} stems", flush=True)

    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write("\n".join(str(x) for x in lines) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
