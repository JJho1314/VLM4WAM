"""OLA: dump planner-predicted SigLIP2 features from the dual-camera K=4 checkpoint, feeding the
model by hand instead of going through the GE-Act dataset.

Why hand-fed: this checkpoint was trained on the GE-Act dual-camera pipeline, whose config points at
a pod path (/root/nas/...) that does not exist on this box. But DualCameraPlannerCollator only needs
four fields per sample -- images (one per camera), prompt, current_camera_images, future_camera_images
-- so the LIBERO mp4s that ARE here can be read directly and shaped into that form. This skips the
whole dataset/config layer while still driving the model exactly as training did.

Frames come from LIBERO lerobot episodes (main + wrist), keyframes at the checkpoint's own offsets.
Features are dumped to disk; the loc-head probe runs locally.
"""
import os, sys, re, json, random, math
from pathlib import Path
import numpy as np, torch, av
from PIL import Image

R = os.environ.get("REPO", "/data/users/junjie/code/VLM4WAM_dual_camera_k4")
for p in ["qwen3_vl_semantic_planner", "qwen3_vl_semantic_planner/dinov3_da3_2b",
          "qwen3_vl_semantic_planner/lingbot_dino_4b", "third_party/FastWAM/src"]:
    d = f"{R}/{p}"
    if os.path.isdir(d): sys.path.insert(0, d)
os.chdir(R); torch.manual_seed(0); random.seed(7)
import train_qwen3vl4b_lingbot_dino_planner as T
from ge_act_dual_camera import DualCameraPlannerCollator
from qwen3vl_wrapper import move_qwen_inputs_to_device, configure_qwen3vl_processor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
dev = "cuda"
def log(m): print(m, flush=True)

DATA = os.environ.get("LIBERO_ROOT", "/data/shared/datasets/libero_fastwam")
SUITES = os.environ.get("SUITES", "spatial,object,goal,10").split(",")
PER_SUITE = int(os.environ.get("PER_SUITE", 6))
EP_PER_TASK = int(os.environ.get("EP_PER_TASK", 1))   # episodes kept per distinct task
CK = Path(os.environ.get("CKPT",
    "outputs/qwen3vl2b_siglip2_da3_libero_dual_camera_k4_wsa_predecoded_b8_restart/step_030000"))
OUT = os.environ.get("OUT", "/data/users/junjie/planner_feats_dualcam_k4.npz")
CAMS = ("observation.images.image", "observation.images.wrist_image")
NOUNS = ["bowl","plate","mug","pot","cabinet","drawer","stove","basket","soup","banana","cheese",
         "cream","ketchup","milk","butter","sauce","tomato","bottle","cup","frying","pan","moka",
         "wine","book","caddy","mustard","box","rack","turkey","salad","dressing",
         # objects that used to fall through the vocabulary, so the destination got picked instead
         "orange","juice","chocolate","pudding","alphabet","ramekin","cookie","microwave","bin",
         "akita","porcelain","white","yellow","black","red"]
# LIBERO instructions read "pick up X ... place it in Y": X is manipulated, Y is only the destination.
# Matching the first vocabulary hit picked Y whenever X was out of vocabulary (e.g. "orange juice"
# -> "basket"), so the probe was asked to localise the wrong object.
_OBJ_RE = re.compile(r"(?:pick up|put|place|push|open|close|turn on)\s+(?:the\s+|both\s+the\s+)?(.+?)"
                     r"\s+(?:and|in|on|to|next|into|from|between|top|of)\b")


def target_nouns(instruction):
    """Nouns of the MANIPULATED object, falling back to a whole-sentence scan."""
    low = instruction.lower()
    m = _OBJ_RE.match(low)
    for scope in ([m.group(1)] if m else []) + [low]:
        hits = [w for w in dict.fromkeys(re.findall(r"[a-z]+", scope)) if w in NOUNS]
        if hits:
            return hits
    return []

meta = json.loads((CK / "planner_meta.json").read_text())
OFFS = [int(x) for x in (meta.get("future_keyframe_offsets") or meta.get("keyframe_offsets"))]
NPREV = int(meta.get("num_previous_frames", 4) or 4)
log(f"ckpt={CK.name} K={meta.get('num_keyframes')} grid={meta.get('grid_size')} "
    f"dim={meta.get('semantic_dim')} cams={meta.get('num_camera_views')} offsets={OFFS}")

proc = configure_qwen3vl_processor(AutoProcessor.from_pretrained(str(CK / "processor"), local_files_only=True))
model = Qwen3VLForConditionalGeneration.from_pretrained(
    str(CK / "qwen3vl_lora_or_model"), dtype=torch.bfloat16,
    attn_implementation="sdpa", local_files_only=True).to(dev).eval()
if hasattr(model.config, "text_config"): model.config.hidden_size = model.config.text_config.hidden_size
model.config.use_cache = False
wrapper = T.PlannerWrapper.from_exported_checkpoint(model=model, checkpoint_dir=CK, metadata=meta)
wrapper.to(dev).eval()
latent_len = int(getattr(wrapper, "latent_len", 0) or meta.get("latent_len") or 0)
coll = DualCameraPlannerCollator(processor=proc, plan_sequence=[f"<|sem_plan_{i}|>" for i in range(latent_len)])
log(f"planner loaded, latent_len={latent_len}")


def read_frames(suite, ei, idxs):
    """frame index -> {camera: HxWx3 uint8} for the requested indices."""
    out = {}
    for cam in CAMS:
        p = f"{DATA}/libero_{suite}_no_noops_lerobot/videos/chunk-000/{cam}/episode_{ei:06d}.mp4"
        c = av.open(p); want, got = set(idxs), {}
        for j, fr in enumerate(c.decode(video=0)):
            if j in want: got[j] = np.asarray(fr.to_ndarray(format="rgb24"))
            if len(got) == len(want): break
        c.close(); out[cam] = got
    return out


def to_chw(a, size=224):
    return torch.from_numpy(np.asarray(Image.fromarray(a).resize((size, size)))).permute(2, 0, 1).float() / 255.0


curs, futs, fp_l, dp_l, prompts, nouns, suites = [], [], [], [], [], [], []
for suite in SUITES:
    eps = [json.loads(l) for l in open(f"{DATA}/libero_{suite}_no_noops_lerobot/meta/episodes.jsonl")]
    seen, got = {}, 0
    for e in eps:
        if got >= PER_SUITE: break
        ei, task = e["episode_index"], e["tasks"][0]
        ws = target_nouns(task)
        if not ws or seen.get(task[:45], 0) >= EP_PER_TASK: continue
        cur_i = NPREV - 1
        kf_i = [cur_i + 1 + o for o in OFFS]
        try: fr = read_frames(suite, ei, [cur_i] + kf_i)
        except Exception: continue
        if any(i not in fr[c] for c in CAMS for i in [cur_i] + kf_i): continue
        seen[task[:45]] = seen.get(task[:45], 0) + 1
        cur_cams = torch.stack([to_chw(fr[c][cur_i]) for c in CAMS])                 # (V,3,H,W)
        fut_cams = torch.stack([torch.stack([to_chw(fr[c][i]) for c in CAMS]) for i in kf_i])  # (K,V,3,H,W)
        sample = {"images": [Image.fromarray(fr[c][cur_i]) for c in CAMS],           # one PIL per camera
                  "prompt": task, "current_camera_images": cur_cams,
                  "future_camera_images": fut_cams, "stem": f"{suite}_{ei}"}
        b = coll([sample])
        for kk in ("stems", "current_camera_images", "future_camera_images"): b.pop(kk, None)
        md = next(wrapper.model.parameters()).dtype
        bi = move_qwen_inputs_to_device(dict(b), dev, model_dtype=md)
        with torch.no_grad():
            fut_dino, fut_depth = wrapper.predict_dino_depth_plan(**bi)
        rs = lambda a: np.asarray(Image.fromarray(a).resize((256, 256)))   # keeps the dump small
        curs.append(np.stack([rs(fr[c][cur_i]) for c in CAMS]).astype(np.uint8))          # (V,256,256,3)
        futs.append(np.stack([[rs(fr[c][i]) for c in CAMS] for i in kf_i]).astype(np.uint8))
        fp_l.append(fut_dino[0].float().cpu().numpy().astype(np.float16))
        dp_l.append(fut_depth[0].float().cpu().numpy().astype(np.float16))
        prompts.append(task); nouns.append("|".join(ws[:3])); suites.append(suite); got += 1
        log(f"[{suite}] {got}/{PER_SUITE} plan{fp_l[-1].shape} [{task[:44]}] {ws[:3]}")

np.savez_compressed(OUT, cur=np.stack(curs), fut=np.stack(futs), fp=np.stack(fp_l), dp=np.stack(dp_l),
                    prompts=np.array(prompts, dtype=object), nouns=np.array(nouns, dtype=object),
                    suites=np.array(suites, dtype=object),
                    meta=np.array(json.dumps({k: meta.get(k) for k in
                        ("num_keyframes", "grid_size", "semantic_dim", "num_camera_views")} |
                        {"offsets": OFFS, "cameras": list(CAMS)}), dtype=object))
log(f"DUMPED {len(curs)} samples -> {OUT} ({os.path.getsize(OUT)//1024//1024} MB)")
log("DUMP-DONE")
