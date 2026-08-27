"""Extract INITIAL (frame-0) observations from LIBERO episodes: main+wrist -> [224,448] composite
+ task instruction. Many episodes per task = diverse initial scenes. Saves libsamples_init.npz."""
import os, json, av, numpy as np
from PIL import Image
ROOT="/data/LFT-W02_data/junjie/data/LIBERO-fastwam"
SUITES=["spatial","object","goal","10"]
PER_TASK=6          # episodes (distinct initial layouts) per task
OUT="/data/LFT-W02_data/junjie/fastwam_sg_ckpt/libsamples_init.npz"

def frame0(mp4):
    c=av.open(mp4)
    for fr in c.decode(video=0):
        a=fr.to_ndarray(format="rgb24"); c.close(); return a
    c.close(); return None
def comp(main_mp4, wrist_mp4):
    m=frame0(main_mp4); w=frame0(wrist_mp4)
    if m is None or w is None: return None
    m=np.asarray(Image.fromarray(m).resize((224,224)))
    w=np.asarray(Image.fromarray(w).resize((224,224)))
    return np.concatenate([m,w],axis=1).astype(np.uint8)  # [224,448] main|wrist

curs,prompts=[],[]
for s in SUITES:
    d=f"{ROOT}/libero_{s}_no_noops_lerobot"
    eps=[json.loads(l) for l in open(f"{d}/meta/episodes.jsonl")]
    by_task={}
    for e in eps: by_task.setdefault(e["tasks"][0], []).append(e["episode_index"])
    imgdir=f"{d}/videos/chunk-000/observation.images.image"
    wrdir=f"{d}/videos/chunk-000/observation.images.wrist_image"
    for task,idxs in by_task.items():
        got=0
        for ei in idxs:
            if got>=PER_TASK: break
            mp4=f"{imgdir}/episode_{ei:06d}.mp4"; wp4=f"{wrdir}/episode_{ei:06d}.mp4"
            if not (os.path.exists(mp4) and os.path.exists(wp4)): continue
            c=comp(mp4,wp4)
            if c is None: continue
            curs.append(c); prompts.append(task); got+=1
    print(f"{s}: {len([p for p in prompts if p in [e['tasks'][0] for e in eps]])} so far, tasks={len(by_task)}",flush=True)
np.savez(OUT, cur=np.stack(curs), prompts=np.array(prompts,dtype=object))
print("SAVED",OUT,len(curs),"init-frame scenes, unique tasks:",len(set(prompts)),flush=True)
