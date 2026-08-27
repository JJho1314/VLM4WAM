"""Stage 1 of the plan-loss A/B: cache a small LIBERO feature set.

Per episode (main camera): frozen SigLIP2 tokens of the CURRENT frame and of K=4 future keyframes
(the plan target), the instruction's SigLIP2 text embedding, plus CLIPSeg target-noun masks on those
keyframes (the localizability supervision used by the honest probe). Output -> data/ab_plan.npz
"""
import os, sys, json, re, math, av, torch, numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.models.clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor
from PIL import Image

dev = "cuda"
ROOT = "/data/LFT-W02_data/junjie/data/LIBERO-fastwam"
SIG = "/data/LFT-W02_data/junjie/weights/siglip2-large-patch16-256"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ab_plan")
CAM = "observation.images.image"
NPREV, KFI, GRID, RES = 4, [0, 3, 5, 8], 16, 256
EP_PER_SUITE = int(os.environ.get("EP_PER_SUITE", 60))
NOUNS = ["bowl","plate","mug","pot","cabinet","drawer","stove","basket","soup","banana","cheese","cream",
         "ketchup","milk","butter","sauce","tomato","bottle","cup","moka","wine","book","mustard","box",
         "pudding","microwave","ramekin","cookie","frypan","pan","alphabet"]
def log(m): print(m, flush=True)


def load_teachers():
    sig = AutoModel.from_pretrained(SIG, dtype=torch.float32).eval().to(dev)
    tok = AutoTokenizer.from_pretrained(SIG)
    cproc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    cseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
    for m in (sig, cseg):
        for p in m.parameters(): p.requires_grad_(False)
    return sig, tok, cproc, cseg


@torch.no_grad()
def siglip_tokens(sig, imgs):
    """imgs [N,H,W,3] uint8 -> patch tokens [N, 256, 1024]. Manual preprocess (repo convention:
    RGB -> [0,1] -> resize -> mean=std=0.5), avoiding processor variants."""
    x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, (RES, RES), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5
    out = sig.vision_model(pixel_values=x).last_hidden_state          # [N, 256, 1024]
    if out.shape[1] == GRID * GRID + 1: out = out[:, 1:]              # tolerate a CLS layout
    return out.float().cpu().numpy().astype(np.float16)


@torch.no_grad()
def siglip_text(sig, tok, s):
    t = tok([s], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(dev)
    o = sig.text_model(**t)
    e = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, -1]
    return F.normalize(e.float(), dim=-1)[0].cpu().numpy().astype(np.float16)


@torch.no_grad()
def clipseg_mask(cproc, cseg, img, noun):
    x = Image.fromarray(img.astype("uint8")).resize((352, 352))
    inp = cproc(text=[noun], images=[x], return_tensors="pt", padding=True)
    inp = {k: v.to(dev) for k, v in inp.items()}
    lg = cseg(**inp).logits
    if lg.ndim == 2: lg = lg[None]
    return F.interpolate(torch.sigmoid(lg[0])[None, None], (GRID, GRID), mode="bilinear")[0, 0].cpu().numpy()


def read_frames(suite, ei, idxs):
    path = f"{ROOT}/libero_{suite}_no_noops_lerobot/videos/chunk-000/{CAM}/episode_{ei:06d}.mp4"
    c = av.open(path); want, frs = set(idxs), {}
    for j, fr in enumerate(c.decode(video=0)):
        if j in want: frs[j] = np.asarray(fr.to_ndarray(format="rgb24"))
        if len(frs) == len(want): break
    c.close(); return frs


def target_noun(task):
    ws = [w for w in re.findall(r"[a-z]+", task.lower()) if w in NOUNS]
    return ws[0] if ws else None


def main():
    sig, tok, cproc, cseg = load_teachers()
    log("teachers loaded")
    cur_l, kf_l, txt_l, msk_l, sid_l = [], [], [], [], []
    cur_idx, kf_idx = NPREV - 1, [NPREV + k for k in KFI]
    for si, suite in enumerate(("object", "spatial", "goal", "10")):
        eps = [json.loads(l) for l in open(f"{ROOT}/libero_{suite}_no_noops_lerobot/meta/episodes.jsonl")]
        kept = 0
        for e in eps:
            if kept >= EP_PER_SUITE: break
            ei, task = e["episode_index"], e["tasks"][0]
            noun = target_noun(task)
            if noun is None: continue
            try: frs = read_frames(suite, ei, [cur_idx] + kf_idx)
            except Exception: continue
            if any(i not in frs for i in [cur_idx] + kf_idx): continue
            feats = siglip_tokens(sig, [frs[i] for i in [cur_idx] + kf_idx])   # [1+K, 256, 1024]
            cur_l.append(feats[0]); kf_l.append(feats[1:])
            txt_l.append(siglip_text(sig, tok, task))
            msk_l.append(np.stack([clipseg_mask(cproc, cseg, frs[i], noun) for i in kf_idx]).astype(np.float16))
            sid_l.append(si); kept += 1
        log(f"{suite}: {kept} episodes")
    os.makedirs(OUT, exist_ok=True)   # one .npy per array: savez_compressed spikes RAM and got OOM-killed
    for name, arr in (("cur", cur_l), ("kf", kf_l), ("txt", txt_l), ("mask", msk_l), ("suite", sid_l)):
        a = np.stack(arr) if name != "suite" else np.array(arr)
        np.save(os.path.join(OUT, f"{name}.npy"), a)
        log(f"  {name}: {a.shape} {a.dtype}")
    log(f"saved {OUT}  N={len(cur_l)}")
    log("ALLDONE")


if __name__ == "__main__":
    main()
