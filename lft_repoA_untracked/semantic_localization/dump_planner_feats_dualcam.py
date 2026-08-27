"""OLA: dump planner-predicted SigLIP2 features from the dual-camera K=4 checkpoint.

Same idea as dump_planner_feats.py (which targeted the single-camera K=1 run) but adapted to
qwen3vl2b_siglip2_da3_libero_dual_camera_k4_wsa_predecoded_b8_restart/step_030000:
  num_keyframes=4, grid_size=16 (256 tokens/keyframe), semantic_dim=1024, num_camera_views=2.

Predictions are dumped to disk; the localisation probe (loc_head.pt) then runs locally, so this
script needs no CLIPSeg and no plotting. Output keys mirror the K=1 dump so the local viz can read
either, with the extra keyframe/camera axes kept explicit.
"""
import os, sys, re, json, random, torch, numpy as np
from pathlib import Path

R = os.environ.get("REPO", "/data/users/junjie/code/VLM4WAM_dual_camera_k4")
for p in ["scripts/qwen3_vl_semantic_planner", "scripts/qwen3_vl_semantic_planner/dinov3_da3_2b",
          "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b",
          "qwen3_vl_semantic_planner", "qwen3_vl_semantic_planner/dinov3_da3_2b"]:
    d = f"{R}/{p}"
    if os.path.isdir(d): sys.path.insert(0, d)
os.chdir(R); torch.manual_seed(0); random.seed(7)
import train_qwen3vl4b_lingbot_dino_planner as T
from qwen3vl_wrapper import move_qwen_inputs_to_device, configure_qwen3vl_processor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
dev = "cuda"
def log(m): print(m, flush=True)

NOUNS = ["bowl","plate","mug","pot","cabinet","drawer","stove","basket","soup","banana","cheese",
         "cream","ketchup","milk","butter","sauce","tomato","bottle","cup","frying","pan","moka",
         "wine","book","caddy","mustard","box","rack","turkey","salad","dressing"]
PER_SUITE = int(os.environ.get("PER_SUITE", 6))
CK = Path(os.environ.get("CKPT",
    "outputs/qwen3vl2b_siglip2_da3_libero_dual_camera_k4_wsa_predecoded_b8_restart/step_030000"))
OUT = os.environ.get("OUT", "/data/users/junjie/planner_feats_dualcam_k4.npz")

meta = json.loads((CK / "planner_meta.json").read_text())
log(f"ckpt={CK}  K={meta.get('num_keyframes')} grid={meta.get('grid_size')} "
    f"dim={meta.get('semantic_dim')} cams={meta.get('num_camera_views')}")
proc = configure_qwen3vl_processor(AutoProcessor.from_pretrained(str(CK / "processor"), local_files_only=True))
model = Qwen3VLForConditionalGeneration.from_pretrained(
    str(CK / "qwen3vl_lora_or_model"), dtype=torch.bfloat16,
    attn_implementation="sdpa", local_files_only=True).to(dev).eval()
if hasattr(model.config, "text_config"): model.config.hidden_size = model.config.text_config.hidden_size
model.config.use_cache = False
wrapper = T.PlannerWrapper.from_exported_checkpoint(model=model, checkpoint_dir=CK, metadata=meta)
wrapper.to(dev).eval()
OFFSETS = [int(x) for x in (meta.get("future_keyframe_offsets") or meta.get("keyframe_offsets"))]
log(f"keyframe offsets from ckpt: {OFFSETS}")
# the plan sequence length is the model's latent_len (training builds it as range(latent_len));
# deriving it from num_task_tokens gave 16 tokens where the model wanted 384
latent_len = int(getattr(wrapper, "latent_len", 0) or getattr(getattr(wrapper, "plan_head", None), "latent_len", 0)
                 or meta.get("latent_len") or 0)
if not latent_len:
    raise RuntimeError(f"cannot determine latent_len; meta keys={sorted(meta)[:40]}")
plan_seq = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
log(f"latent_len={latent_len} -> {len(plan_seq)} plan tokens")
coll = T.Collator(processor=proc, plan_sequence=plan_seq)
log("planner loaded")

curs, futs, fp_l, cp_l, prompts, nouns, suites = [], [], [], [], [], [], []
# use the SAME env var the training launcher sets, so the dataset loads in dual-camera mode
CFG = Path(os.environ.get("FASTWAM_DATA_CONFIG",
                          "third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml"))
log(f"data config: {CFG}")
for sdir in os.environ["FASTWAM_DATASET_DIRS"].split(":"):
    suite = Path(sdir).name.replace("libero_", "").replace("_no_noops_lerobot", "")
    ds = T.FastWAMOnlinePlannerDataset.from_config(
        CFG, dataset_dirs=[sdir],
        text_embedding_cache_dir=Path(os.environ["FASTWAM_TEXT_EMBEDDING_CACHE_DIR"]),
        pretrained_norm_stats=Path(os.environ["FASTWAM_PRETRAINED_NORM_STATS"]),
        max_samples=0, offsets=OFFSETS)              # read from planner_meta so it matches training
    seen, got = set(), 0
    for _ in range(6000):
        if got >= PER_SUITE: break
        s = ds[random.randrange(len(ds))]
        ins = str(s.get("instruction") or s.get("prompt") or "")
        ws = [w for w in dict.fromkeys(re.findall(r"[a-z]+", ins.lower())) if w in NOUNS]
        k = ins.strip()[:45]
        if not ws or k in seen: continue
        seen.add(k)
        kfs = np.asarray(s["keyframe_images"]); cur = np.asarray(s["current_image"])
        b = coll([s])
        for kk in ("stems", "keyframe_images", "current_image", "future_video_effective_fps"): b.pop(kk, None)
        md = next(wrapper.model.parameters()).dtype
        bi = move_qwen_inputs_to_device(dict(b), dev, model_dtype=md)
        with torch.no_grad():
            # this checkpoint has use_current_alignment=False (future-only), so the current-plan API
            # refuses to run; predict_dino_depth_plan handles both modes and returns the future plan
            fut_dino, fut_depth = wrapper.predict_dino_depth_plan(**bi)
        curs.append(cur.astype(np.uint8))
        futs.append(kfs.astype(np.uint8))                      # (K,V,H,W,3) or (K,H,W,3)
        fp_l.append(fut_dino[0].float().cpu().numpy().astype(np.float16))
        cp_l.append(fut_depth[0].float().cpu().numpy().astype(np.float16))   # depth plan, not current
        prompts.append(ins); nouns.append("|".join(ws[:3])); suites.append(suite); got += 1
        log(f"[{suite}] {got}/{PER_SUITE} fut{fp_l[-1].shape} depth{cp_l[-1].shape} [{ins[:40]}] {ws[:3]}")
    del ds

np.savez_compressed(OUT, cur=np.stack(curs), fut=np.stack(futs), fp=np.stack(fp_l), cp=np.stack(cp_l),
                    prompts=np.array(prompts, dtype=object), nouns=np.array(nouns, dtype=object),
                    suites=np.array(suites, dtype=object),
                    meta=np.array(json.dumps({k: meta.get(k) for k in
                        ("num_keyframes", "grid_size", "semantic_dim", "num_camera_views")}), dtype=object))
log(f"DUMPED {len(curs)} samples -> {OUT} ({os.path.getsize(OUT)//1024//1024} MB)")
log("DUMP-DONE")
