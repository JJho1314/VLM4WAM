"""Select ~half of the IoU>=0.5 ('completely correct') episodes, maximizing task
diversity (round-robin over normalized captions, and within a task over
episodes/camera-views), then build a training-ready dataset dir of hardlinks."""

import os
import re
import glob
from collections import OrderedDict, defaultdict

DS = "/data/user/jhe724/workspace/datasets"
FULL = f"{DS}/droid_success_v21_target_aware_left_right_480x864"
EVAL = f"{FULL}/instructsam_mask_eval"
OUT_DIR = f"{DS}/droid_v21_iou50_taskdiverse_half"
SUBDIRS = ["videos", "masks", "metas", "target_features_ft", "target_features_rawseg_ft"]
EXT = {"videos": ".mp4", "masks": ".npz", "metas": ".txt",
       "target_features_ft": ".pt", "target_features_rawseg_ft": ".pt"}

PREFIX = re.compile(r"^a franka robotic arm with a parallel-jaw gripper\s*", re.I)
EP = re.compile(r"^(episode_\d+)_(.+)$")


def norm_caption(stem):
    p = f"{FULL}/metas/{stem}.txt"
    try:
        cap = open(p).read().strip()
    except Exception:
        return None
    c = cap.lower().replace("[tgt]", " ")
    c = PREFIX.sub("", c)
    c = re.sub(r"[^a-z0-9 ]", " ", c)
    c = re.sub(r"\s+", " ", c).strip()
    return c or "<empty>"


def main():
    stems = [s for s in open(f"{EVAL}/mask_correct_stems_iou50.txt").read().split() if s]
    print(f"correct (IoU>=0.5) stems: {len(stems)}", flush=True)
    target = len(stems) // 2
    print(f"target half: {target}", flush=True)

    # group by normalized caption (task)
    task_groups = defaultdict(list)
    skipped = 0
    for s in stems:
        c = norm_caption(s)
        if c is None:
            skipped += 1
            continue
        task_groups[c].append(s)
    print(f"distinct tasks: {len(task_groups)} | meta-missing: {skipped}", flush=True)

    # within each task, order stems to spread episodes & camera views:
    # round A = first view of every episode, round B = second view, ...
    ordered_groups = OrderedDict()
    for task in sorted(task_groups):
        by_ep = defaultdict(list)
        for s in sorted(task_groups[task]):
            m = EP.match(s)
            ep = m.group(1) if m else s
            by_ep[ep].append(s)
        max_views = max(len(v) for v in by_ep.values())
        order = []
        for k in range(max_views):
            for ep in sorted(by_ep):
                if k < len(by_ep[ep]):
                    order.append(by_ep[ep][k])
        ordered_groups[task] = order

    # global round-robin across tasks: take 1 per task per round
    selected = []
    round_idx = 0
    tasks = sorted(ordered_groups)
    while len(selected) < target:
        progressed = False
        for task in tasks:
            grp = ordered_groups[task]
            if round_idx < len(grp):
                selected.append(grp[round_idx])
                progressed = True
                if len(selected) >= target:
                    break
        if not progressed:
            break
        round_idx += 1
    print(f"selected: {len(selected)} | rounds used: {round_idx + 1}", flush=True)

    # diversity report
    sel_tasks = {norm_caption(s) for s in selected}
    print(f"distinct tasks IN selection: {len(sel_tasks)} "
          f"({len(sel_tasks)/len(selected)*100:.1f}% unique-task ratio)", flush=True)

    # build dataset dir of hardlinks
    for sub in SUBDIRS:
        os.makedirs(f"{OUT_DIR}/{sub}", exist_ok=True)
    linked = defaultdict(int)
    missing = defaultdict(int)
    for s in selected:
        for sub in SUBDIRS:
            src = f"{FULL}/{sub}/{s}{EXT[sub]}"
            dst = f"{OUT_DIR}/{sub}/{s}{EXT[sub]}"
            if os.path.exists(src):
                if not os.path.exists(dst):
                    os.link(src, dst)
                linked[sub] += 1
            else:
                missing[sub] += 1
        # link the mask preview png too (harmless, useful for inspection)
        mp = f"{FULL}/target_features_ft/{s}_mask.png"
        if os.path.exists(mp):
            dmp = f"{OUT_DIR}/target_features_ft/{s}_mask.png"
            if not os.path.exists(dmp):
                os.link(mp, dmp)

    print("linked:", dict(linked), flush=True)
    print("missing:", dict(missing), flush=True)
    open(f"{OUT_DIR}/selected_stems.txt", "w").write("\n".join(sorted(selected)) + "\n")
    # empty exclude file so the loader's "auto" path is explicit/clean
    open(f"{OUT_DIR}/exclude_no_tgt_stems.txt", "w").write("")
    print("DONE", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
