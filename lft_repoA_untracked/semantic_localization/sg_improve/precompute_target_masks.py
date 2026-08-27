"""Offline supervision for the auxiliary spatial-grounding loss (Option 2).
For each LIBERO episode: extract the K future keyframes ([0,3,5,8]) x 2 views, compute a CLIPSeg soft
pseudo-GT mask of the instruction's TARGET NOUN on each keyframe (16x16 to match SigLIP patch grid),
plus the noun's CLIP-text embedding. Saves per-episode -> target_masks[V,K,16,16], target_noun_emb[512].
Feed these into the training batch as batch['target_masks'] / batch['target_noun_emb'] (see INTEGRATION).

MUST align with the trainer's keyframe indices & view order. Adjust KFI / camera order to match your
GEActDualCameraPlannerDataset config (num_camera_views, future_keyframe_offsets)."""
import os, sys, json, re, av, torch, numpy as np
from PIL import Image
import torch.nn.functional as F
from transformers.models.clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor
dev = "cuda"
ROOT = "/data/LFT-W02_data/junjie/data/LIBERO-fastwam"
OUT = "/data/LFT-W02_data/junjie/data/LIBERO-target-masks"
NPREV = 4; KFI = [0, 3, 5, 8]; GRID = 16   # match planner tokens_per_keyframe=256 -> 16x16
CAMS = ("observation.images.image", "observation.images.wrist_image")
NOUNS = ["bowl","plate","mug","pot","cabinet","drawer","stove","basket","soup","banana","cheese","cream",
         "ketchup","milk","butter","sauce","tomato","bottle","cup","moka","wine","book","mustard","box",
         "pudding","microwave","ramekin","cookie","frypan","pan","alphabet"]
os.makedirs(OUT, exist_ok=True)
cproc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
cseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
for p in cseg.parameters(): p.requires_grad_(False)

@torch.no_grad()
def noun_emb(noun):
    inp = cproc(text=[f"a photo of a {noun}"], return_tensors="pt", padding="max_length", max_length=77)
    inp = {k: v.to(dev) for k, v in inp.items() if k in ("input_ids", "attention_mask")}
    o = cseg.clip.text_model(**inp); t = o.pooler_output if hasattr(o, "pooler_output") else o[1]
    return F.normalize(t.float(), dim=-1)[0].cpu().numpy()
@torch.no_grad()
def clipseg_mask(img, noun):
    x = Image.fromarray(img.astype("uint8")).resize((352, 352))
    inp = cproc(text=[noun], images=[x], return_tensors="pt", padding=True); inp = {k: v.to(dev) for k, v in inp.items()}
    lg = cseg(**inp).logits
    if lg.ndim == 2: lg = lg[None]
    return F.interpolate(torch.sigmoid(lg[0])[None, None], (GRID, GRID), mode="bilinear")[0, 0].cpu().numpy()

def read_frames(suite, ei, idxs):
    d = f"{ROOT}/libero_{suite}_no_noops_lerobot/videos/chunk-000"; out = {}
    for cam in CAMS:
        c = av.open(f"{d}/{cam}/episode_{ei:06d}.mp4"); want = set(idxs); frs = {}
        for j, fr in enumerate(c.decode(video=0)):
            if j in want: frs[j] = np.asarray(fr.to_ndarray(format="rgb24"))
            if len(frs) == len(want): break
        c.close(); out[cam] = frs
    return out

def target_noun(task):
    ws = [w for w in re.findall(r"[a-z]+", task.lower()) if w in NOUNS]
    return ws[0] if ws else None

def main():
    for suite in ("object", "spatial", "goal", "10"):
        eps = [json.loads(l) for l in open(f"{ROOT}/libero_{suite}_no_noops_lerobot/meta/episodes.jsonl")]
        for e in eps:
            ei, task = e["episode_index"], e["tasks"][0]
            noun = target_noun(task)
            if noun is None: continue
            abs_idx = [NPREV + k for k in KFI]           # future keyframes in absolute frame index
            fr = read_frames(suite, ei, abs_idx)
            masks = np.zeros((len(CAMS), len(KFI), GRID, GRID), np.float16)
            ok = True
            for vi, cam in enumerate(CAMS):
                for ki, ai in enumerate(abs_idx):
                    if ai not in fr[cam]: ok = False; break
                    masks[vi, ki] = clipseg_mask(fr[cam][ai], noun)
            if not ok: continue
            np.savez(f"{OUT}/{suite}_{ei:06d}.npz", target_masks=masks, target_noun_emb=noun_emb(noun).astype(np.float16), noun=noun)
        print(f"{suite}: done", flush=True)
    print("ALLDONE", flush=True)

if __name__ == "__main__":
    main()
