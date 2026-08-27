"""POD: load cur_k1/step_030000 planner, predict future/current SigLIP features on noun-bearing
LIBERO samples ACROSS ALL 4 SUITES (spatial/object/goal/10), dump {preds + frames + prompt +
suite} to persistent disk. Localization done locally with the trained loc-head."""
import os, sys, re, json, random, torch, numpy as np
from pathlib import Path
R = "/data/users/junjie/code/VLM4WAM_k1_zero2_bidir"
for p in ["scripts/qwen3_vl_semantic_planner","scripts/qwen3_vl_semantic_planner/dinov3_da3_2b","scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"]:
    sys.path.insert(0, f"{R}/{p}")
os.chdir(R); torch.manual_seed(0); random.seed(7)
import train_qwen3vl4b_lingbot_dino_planner as T
from qwen3vl_wrapper import move_qwen_inputs_to_device, configure_qwen3vl_processor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
dev = "cuda"
def log(m): print(m, flush=True)

NOUNS=["bowl","plate","mug","pot","cabinet","drawer","stove","basket","soup","banana","cheese",
       "cream","ketchup","milk","butter","sauce","tomato","bottle","cup","frying","pan","moka",
       "wine","book","caddy","mustard","box","rack","turkey","salad","dressing"]
PER_SUITE = 4  # distinct noun-bearing tasks per suite

CK=Path("outputs/qwen3vl2b_siglip2_da3_libero_cur_k1/step_030000")
meta=json.loads((CK/"planner_meta.json").read_text())
proc=configure_qwen3vl_processor(AutoProcessor.from_pretrained(str(CK/"processor"),local_files_only=True))
model=Qwen3VLForConditionalGeneration.from_pretrained(str(CK/"qwen3vl_lora_or_model"),dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).to(dev).eval()
if hasattr(model.config,"text_config"): model.config.hidden_size=model.config.text_config.hidden_size
model.config.use_cache=False
wrapper=T.PlannerWrapper.from_exported_checkpoint(model=model,checkpoint_dir=CK,metadata=meta); wrapper.to(dev).eval()
plan_seq=[f"<|sem_plan_{i}|>" for i in range((4 if meta.get("independent_modality_task_tokens") else 2)*int(meta["num_task_tokens"]))]
coll=T.Collator(processor=proc,plan_sequence=plan_seq)
log("planner loaded")

curs,futs,fp_l,cp_l,prompts,nouns,suites=[],[],[],[],[],[],[]
CFG=Path("third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml")
for sdir in os.environ["FASTWAM_DATASET_DIRS"].split(":"):
    suite=Path(sdir).name.replace("libero_","").replace("_no_noops_lerobot","")
    ds=T.FastWAMOnlinePlannerDataset.from_config(CFG, dataset_dirs=[sdir],
        text_embedding_cache_dir=Path(os.environ["FASTWAM_TEXT_EMBEDDING_CACHE_DIR"]),
        pretrained_norm_stats=Path(os.environ["FASTWAM_PRETRAINED_NORM_STATS"]),max_samples=0,offsets=[8])
    seen=set(); got=0
    for _ in range(6000):
        if got>=PER_SUITE: break
        s=ds[random.randrange(len(ds))]; ins=str(s.get("instruction") or s.get("prompt") or "")
        ws=[w for w in dict.fromkeys(re.findall(r"[a-z]+",ins.lower())) if w in NOUNS]
        k=ins.strip()[:45]
        if not ws or k in seen: continue
        seen.add(k)
        kfs=np.asarray(s["keyframe_images"]); cur=np.asarray(s["current_image"]); fut=np.asarray(kfs[0])
        b=coll([s]); [b.pop(kk,None) for kk in ("stems","keyframe_images","current_image","future_video_effective_fps")]
        md=next(wrapper.model.parameters()).dtype; bi=move_qwen_inputs_to_device(dict(b),dev,model_dtype=md)
        with torch.no_grad(): pl=wrapper.predict_current_future_plans(**bi)
        curs.append(cur.astype(np.uint8)); futs.append(fut.astype(np.uint8))
        fp_l.append(pl["future_dino"][0].float().cpu().numpy().astype(np.float16))
        cp_l.append(pl["current_dino"][0].float().cpu().numpy().astype(np.float16))
        prompts.append(ins); nouns.append("|".join(ws[:3])); suites.append(suite); got+=1
        log(f"[{suite}] {got}/{PER_SUITE} [{ins[:42]}] {ws[:3]}")
    del ds
out="/data/users/junjie/planner_feats_suites.npz"
np.savez_compressed(out, cur=np.stack(curs), fut=np.stack(futs), fp=np.stack(fp_l), cp=np.stack(cp_l),
    prompts=np.array(prompts,dtype=object), nouns=np.array(nouns,dtype=object), suites=np.array(suites,dtype=object))
log(f"DUMPED {len(curs)} across suites -> {os.path.getsize(out)//1024//1024} MB")
