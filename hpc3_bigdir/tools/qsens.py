import torch, numpy as np, imageio.v3 as iio
from PIL import Image
BASE="/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets"
CD=f"{BASE}/robointer_74616_yellow_carrot_prompt_targetaware_dataset"
BD=f"{BASE}/robointer_74616_banana_prompt_targetaware_dataset"
def dense(p): return torch.load(p,map_location="cpu",weights_only=False)["target_feature"].float()
carrot=dense(f"{CD}/target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt_s20260613/74616_exterior_image_1_left.pt")
banana=dense(f"{BD}/target_features_instructsam_decoder_dense_stage2_lora_banana_prompt_s20260613/74616_exterior_image_1_left.pt")
g=int(round(carrot.shape[0]**0.5))
C=carrot.view(g,g,-1); B=banana.view(g,g,-1)
# per-cell L2 diff and cosine (query-sensitivity map), non-circular
diff=(C-B).norm(dim=-1)                                   # g,g
Cn=torch.nn.functional.normalize(C,dim=-1); Bn=torch.nn.functional.normalize(B,dim=-1)
celldist=1-(Cn*Bn).sum(-1)                                # 0 identical .. larger=differs
print("global cosine:", round(float(torch.nn.functional.cosine_similarity(carrot.flatten(),banana.flatten(),dim=0)),4))
print(f"per-cell cosine-dist: mean={celldist.mean():.4f} max={celldist.max():.4f} >0.02 cells={(celldist>0.02).sum().item()}/{g*g}")
# carrot mask -> 32x32
z=np.load(f"{CD}/target_masks/74616_exterior_image_1_left.npz",allow_pickle=True); arr=z[z.files[0]]
while arr.ndim>2: arr=arr.max(0)
H,W=arr.shape; mask=(arr>0).astype(np.float32)
cm=torch.nn.functional.adaptive_avg_pool2d(torch.tensor(mask).view(1,1,H,W),(g,g)).view(g,g)
cmb=(cm>0.25)
# does the query-induced change concentrate in the carrot region?
din=float(diff[cmb].mean()); dout=float(diff[~cmb].mean())
print(f"query-change(diff) inside carrot mask={din:.3f}  outside={dout:.3f}  ratio={din/max(dout,1e-6):.2f}x")
# carrot prototype heatmap
proto=torch.nn.functional.normalize((C*cm.unsqueeze(-1)).sum((0,1))/cm.sum(),dim=0)
hc=(Cn*proto).sum(-1); print(f"carrot-proto heatmap: inside={float(hc[cmb].mean()):.3f} outside={float(hc[~cmb].mean()):.3f} lift={float(hc[cmb].mean()-hc[~cmb].mean()):+.3f}")

# ---- render figure ----
fr=iio.imread(f"{CD}/videos/74616_exterior_image_1_left.mp4", index=0)   # H,W,3
fH,fW=fr.shape[:2]
def up(t):
    t=(t-t.min())/(t.max()-t.min()+1e-6)
    im=Image.fromarray((t.numpy()*255).astype(np.uint8)).resize((fW,fH),Image.BILINEAR)
    return np.asarray(im).astype(np.float32)/255
def overlay(base,heat,color=(255,40,20)):
    out=base.astype(np.float32).copy()
    a=(heat[...,None])*0.55
    col=np.array(color,np.float32)
    return (out*(1-a)+col*a).clip(0,255).astype(np.uint8)
maskbig=np.asarray(Image.fromarray((mask*255).astype(np.uint8)).resize((fW,fH),Image.NEAREST)).astype(np.float32)/255
panels=[
  ("source", fr.astype(np.uint8)),
  ("carrot GT mask", overlay(fr,maskbig,(40,200,60))),
  ("query-change (carrot vs banana)", overlay(fr,up(diff))),
  ("carrot query.dense heatmap", overlay(fr,up(hc))),
]
pad=6; strip=np.ones((fH+24, fW*4+pad*5, 3),np.uint8)*255
from PIL import ImageDraw
img=Image.fromarray(strip); dr=ImageDraw.Draw(img)
for i,(name,p) in enumerate(panels):
    x=pad+i*(fW+pad); img.paste(Image.fromarray(p),(x,20)); dr.text((x,4),name,fill=(0,0,0))
img.save("/data/user/jhe724/workspace/VLM4WAM/tools/qsens_figure.png")
print("saved figure -> tools/qsens_figure.png  frame=",fH,"x",fW)
