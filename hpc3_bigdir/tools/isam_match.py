import os, sys, json, tempfile
import numpy as np, torch
from PIL import Image, ImageDraw

REPO=os.environ["REPO_ROOT"]; SRC=os.environ["INSTRUCTSAM_SOURCE_ROOT"]; MODEL=os.environ["INSTRUCTSAM_MODEL_PATH"]
for p in (f"{REPO}/scripts/_env_stubs", REPO, SRC):
    if p not in sys.path: sys.path.insert(0,p)
from cosmos_predict2._src.predict2.target_aware.instructsam_mask import (
    InstructSAMTargetMaskGenerator, read_first_frame_image)
from instructsam import mm_infer_segmentation

OUT="/data/user/jhe724/workspace/VLM4WAM/feature_guidance_analysis/instructsam_query_match_74616"
os.makedirs(OUT, exist_ok=True)
VID="/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets/robointer_74616_yellow_carrot_prompt_targetaware_dataset/videos/74616_exterior_image_1_left.mp4"
QUERIES={
 "carrot":"Please segment the yellow carrot with green leaves in the image.",
 "banana":"Please segment the yellow banana in the image.",
}
print("loading InstructSAM...", flush=True)
gen=InstructSAMTargetMaskGenerator(MODEL, source_root=SRC)
img=read_first_frame_image(VID)              # PIL RGB
tmp=os.path.join(OUT,"source_frame.png"); img.save(tmp)
W,H=img.size; print("image size",W,"x",H, flush=True)

def soft_best(query):
    conv=[{"role":"user","content":[{"type":"image","image":tmp},{"type":"text","text":query}]}]
    output,pred_masks,cls_score=mm_infer_segmentation(tmp, gen.processor, conv, gen.model, gen.tokenizer)
    print(f"  query={query!r}\n   output={str(output)[:120]!r}", flush=True)
    if pred_masks is None:
        print("   pred_masks=None"); return None,None
    pm=pred_masks.detach().float().cpu()
    print("   pred_masks shape",tuple(pm.shape),"range",float(pm.min()),float(pm.max()),
          "cls_score",None if cls_score is None else tuple(cls_score.shape), flush=True)
    if pm.ndim==4: pm=pm[0]                    # (N,H,W) drop batch
    if pm.ndim==2: pm=pm[None]
    N=pm.shape[0]
    cs=(cls_score.detach().float().cpu().reshape(-1)[:N] if cls_score is not None else torch.zeros(N))
    peaks=pm.amax(dim=(1,2))
    for i in range(N):
        print(f"     slot{i}: cls={float(cs[i]):+.3f} peak_logit={float(peaks[i]):+.3f}", flush=True)
    best=int(cs.argmax())
    print(f"   -> selected slot {best} (cls={float(cs[best]):+.3f} peak={float(peaks[best]):+.3f})", flush=True)
    raw=pm[best]                              # H,W logits
    soft=torch.sigmoid(raw)
    return raw.numpy(), soft.numpy()

res={}
for name,q in QUERIES.items():
    print(f"\n=== {name} ===", flush=True)
    raw,soft=soft_best(q)
    res[name]=soft
    if soft is not None:
        # resize to image size + 32x32 conditioning grid
        s=torch.tensor(soft)[None,None]
        full=torch.nn.functional.interpolate(s,size=(H,W),mode="bilinear",align_corners=False)[0,0].numpy()
        g32=torch.nn.functional.interpolate(s,size=(32,32),mode="bilinear",align_corners=False)[0,0].numpy()
        np.save(f"{OUT}/{name}_softmap_full.npy", full)
        np.save(f"{OUT}/{name}_where_32x32.npy", g32)
        res[name]=full

# discriminability + figure
def norm(x):
    x=np.asarray(x,np.float32); return (x-x.min())/(x.max()-x.min()+1e-6)
if res.get("carrot") is not None and res.get("banana") is not None:
    a=norm(res["carrot"]).ravel()-norm(res["carrot"]).mean()
    b=norm(res["banana"]).ravel()-norm(res["banana"]).mean()
    corr=float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6))
    ca=res["carrot"]>0.5; ba=res["banana"]>0.5
    iou=float((ca&ba).sum()/max((ca|ba).sum(),1))
    cyx=np.unravel_index(np.argmax(res["carrot"]),res["carrot"].shape)
    byx=np.unravel_index(np.argmax(res["banana"]),res["banana"].shape)
    print(f"\n=== DISCRIMINABILITY ===")
    print(f"spatial corr(carrot_soft, banana_soft) = {corr:.4f}")
    print(f"IoU(carrot>0.5, banana>0.5) = {iou:.4f}")
    print(f"carrot peak (y,x)={cyx}  banana peak (y,x)={byx}")
    json.dump({"spatial_corr":corr,"iou_thresh0.5":iou,
               "carrot_peak_yx":list(map(int,cyx)),"banana_peak_yx":list(map(int,byx))},
              open(f"{OUT}/discriminability.json","w"),indent=2)
    base=np.asarray(img).astype(np.float32)
    def ov(h,color):
        a=norm(h)[...,None]*0.6
        return (base*(1-a)+np.array(color,np.float32)*a).clip(0,255).astype(np.uint8)
    panels=[("source",np.asarray(img)),
            ("carrot query -> softmap",ov(res["carrot"],(255,40,20))),
            ("banana query -> softmap",ov(res["banana"],(255,210,0))),
            ("abs diff",ov(np.abs(norm(res["carrot"])-norm(res["banana"])),(0,180,255)))]
    pad=8; strip=Image.new("RGB",(W*4+pad*5,H+28),(255,255,255)); dr=ImageDraw.Draw(strip)
    for i,(nm,p) in enumerate(panels):
        x=pad+i*(W+pad); strip.paste(Image.fromarray(p),(x,24)); dr.text((x,6),nm,fill=(0,0,0))
    strip.save(f"{OUT}/query_match_compare.png")
    print("saved figure ->",f"{OUT}/query_match_compare.png")
print("\nDONE")
