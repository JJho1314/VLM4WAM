import os,sys,random,numpy as np
from pathlib import Path
R="/data/users/junjie/code/VLM4WAM_k1_zero2_bidir"
for p in ["scripts/qwen3_vl_semantic_planner","scripts/qwen3_vl_semantic_planner/dinov3_da3_2b","scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"]:
    sys.path.insert(0,f"{R}/{p}")
os.chdir(R); random.seed(11)
import train_qwen3vl4b_lingbot_dino_planner as T
ds=T.FastWAMOnlinePlannerDataset.from_config(Path("third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml"),
  dataset_dirs=os.environ["FASTWAM_DATASET_DIRS"].split(":"),
  text_embedding_cache_dir=Path(os.environ["FASTWAM_TEXT_EMBEDDING_CACHE_DIR"]),
  pretrained_norm_stats=Path(os.environ["FASTWAM_PRETRAINED_NORM_STATS"]),max_samples=0,offsets=[8])
N=450
curs,futs,prompts=[],[],[]
for i in range(N):
    c=ds[random.randrange(len(ds))]
    curs.append(np.asarray(c["current_image"]).astype(np.uint8))
    futs.append(np.asarray(c["keyframe_images"])[0].astype(np.uint8))
    prompts.append(str(c.get("instruction") or c.get("prompt") or ""))
out="/data/users/junjie/libsamples.npz"
np.savez_compressed(out,cur=np.stack(curs),fut=np.stack(futs),prompts=np.array(prompts,dtype=object))
print("DUMPED",N,"->",os.path.getsize(out)//1024//1024,"MB",flush=True)
