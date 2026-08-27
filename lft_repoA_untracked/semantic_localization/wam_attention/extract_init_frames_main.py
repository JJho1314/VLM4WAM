"""Main-camera-only initial frames (no wrist composite). frame-0 of many episodes/task -> [224,224]
main view + task. More episodes per task = more scenes. Saves libsamples_init_main.npz."""
import os, json, av, numpy as np
from PIL import Image
ROOT = "/data/LFT-W02_data/junjie/data/LIBERO-fastwam"
SUITES = ["spatial", "object", "goal", "10"]
PER_TASK = int(os.environ.get("PER_TASK", 12))     # episodes (distinct initial layouts) per task
OUT = "/data/LFT-W02_data/junjie/fastwam_sg_ckpt/libsamples_init_main.npz"


def frame0(mp4):
    c = av.open(mp4)
    for fr in c.decode(video=0):
        a = fr.to_ndarray(format="rgb24"); c.close(); return a
    c.close(); return None


curs, prompts = [], []
for s in SUITES:
    d = f"{ROOT}/libero_{s}_no_noops_lerobot"
    eps = [json.loads(l) for l in open(f"{d}/meta/episodes.jsonl")]
    by_task = {}
    for e in eps: by_task.setdefault(e["tasks"][0], []).append(e["episode_index"])
    imgdir = f"{d}/videos/chunk-000/observation.images.image"
    for task, idxs in by_task.items():
        got = 0
        for ei in idxs:
            if got >= PER_TASK: break
            mp4 = f"{imgdir}/episode_{ei:06d}.mp4"
            if not os.path.exists(mp4): continue
            m = frame0(mp4)
            if m is None: continue
            curs.append(np.asarray(Image.fromarray(m).resize((224, 224))).astype(np.uint8))
            prompts.append(task); got += 1
    print(f"{s}: total {len(curs)} scenes so far, tasks={len(by_task)}", flush=True)
np.savez(OUT, cur=np.stack(curs), prompts=np.array(prompts, dtype=object))
print("SAVED", OUT, len(curs), "main-only scenes, unique tasks:", len(set(prompts)), flush=True)
