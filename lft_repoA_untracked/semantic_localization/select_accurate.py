"""LOCAL: score every candidate by how well the planner-predicted-feature localization lands on
the target object (peak of the loc-head heatmap vs CLIPSeg pseudo-GT mask, on current+future
frames), rank, and render the TOP-K most accurate examples only. Target = first noun."""
import os, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
dev="cuda"
P="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization"
OUTDIR=f"{P}/figs"; HEADDIR=P; TOPK=8
from transformers.models.clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor
cproc=CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
cseg=CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
for p in cseg.parameters(): p.requires_grad_(False)
print("clipseg loaded",flush=True)

@torch.no_grad()
def temb(w):
    inp=cproc(text=[f"a photo of a {w}"],return_tensors="pt",padding="max_length",max_length=77)
    inp={k:v.to(dev) for k,v in inp.items() if k in ("input_ids","attention_mask")}
    o=cseg.clip.text_model(**inp); t=o.pooler_output if hasattr(o,"pooler_output") else o[1]
    return F.normalize(t.float(),dim=-1)
@torch.no_grad()
def cmask(comp,w):  # CLIPSeg pseudo-GT mask at 16x16 (0..1)
    img=Image.fromarray(comp.astype("uint8")).resize((352,352))
    inp=cproc(text=[w],images=[img],return_tensors="pt",padding=True); inp={k:v.to(dev) for k,v in inp.items()}
    lg=cseg(**inp).logits
    if lg.ndim==2: lg=lg[None]
    return F.interpolate(torch.sigmoid(lg[0])[None,None],(16,16),mode="bilinear")[0,0].cpu().numpy()

class LocHead(nn.Module):
    def __init__(self,d=1024,hid=256,tdim=512):
        super().__init__(); self.film=nn.Linear(tdim,2*hid); self.inp=nn.Conv2d(d,hid,1)
        self.net=nn.Sequential(nn.Conv2d(hid,hid,3,padding=1),nn.GroupNorm(8,hid),nn.GELU(),nn.Conv2d(hid,hid,3,padding=1),nn.GroupNorm(8,hid),nn.GELU())
        self.out=nn.Conv2d(hid,1,1)
    def forward(self,f,t):
        B=f.shape[0]; x=f.permute(0,2,1).reshape(B,-1,16,16); x=self.inp(x)
        g,b=self.film(t).chunk(2,-1); x=x*(1+g[...,None,None])+b[...,None,None]
        return self.out(self.net(x)).squeeze(1)
head=LocHead().to(dev); head.load_state_dict(torch.load(f"{HEADDIR}/loc_head.pt")["state"]); head.eval()
print("loc-head loaded",flush=True)

d=np.load(f"{P}/data/planner_feats_suites.npz",allow_pickle=True)
CUR,FUT,FP,CP=d["cur"],d["fut"],d["fp"],d["cp"]
PROMPTS=[str(x) for x in d["prompts"]]; NOUNS=[str(x).split("|") for x in d["nouns"]]
SUITES=[str(x) for x in d["suites"]] if "suites" in d else ["?"]*len(CUR)
print("candidates:",len(CUR),flush=True)

@torch.no_grad()
def hmap(feat_np,w):
    f=torch.from_numpy(feat_np.astype(np.float32)).to(dev)
    return torch.sigmoid(head(f[None],temb(w))[0]).cpu().numpy()  # 16x16
def peak_score(hm,mask):
    # CLIPSeg mask value at the heatmap's peak (0..1) + soft-IoU, averaged
    yx=np.unravel_index(np.argmax(hm),hm.shape); pk=float(mask[yx])
    hb=hm>np.percentile(hm,90); mb=mask>0.4
    iou=(hb&mb).sum()/((hb|mb).sum()+1e-6)
    return 0.7*pk+0.3*iou

scores=[]
for ti in range(len(CUR)):
    w=NOUNS[ti][0]
    mk_c=cmask(CUR[ti],w); mk_f=cmask(FUT[ti],w)
    hc=hmap(CP[ti],w); hf=hmap(FP[ti],w)
    sc=0.5*peak_score(hc,mk_c)+0.5*peak_score(hf,mk_f)
    scores.append((sc,ti))
    print(f"score={sc:.3f} [{SUITES[ti]}] '{w}' {PROMPTS[ti][:42]}",flush=True)
scores.sort(reverse=True)
top=[ti for _,ti in scores[:TOPK]]
print("=== TOP", TOPK, "===", [(round(s,3),SUITES[ti],NOUNS[ti][0]) for s,ti in scores[:TOPK]],flush=True)

# render top-K only (clear previous planner figs first is done by caller)
TURBO=cm.get_cmap("turbo")
def to448(a):
    t=torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1).unsqueeze(0).float()
    return F.interpolate(t,(224,448),mode="bilinear",align_corners=False)[0].permute(1,2,0).clamp(0,1).numpy()
def sharpen(hm,lo=55.0,hi=99.5,gamma=1.9):
    plo,phi=np.percentile(hm,[lo,hi]); x=np.clip((hm-plo)/(phi-plo+1e-6),0,1); return x**gamma
def ov(feat_np,w,base):
    hm=hmap(feat_np,w); up=to448(TURBO(sharpen(hm))[...,:3]); return base*0.45+up*0.55

for rank,ti in enumerate(top):
    w=NOUNS[ti][0]; ins=PROMPTS[ti]; suite=SUITES[ti]; sc=dict((t,s) for s,t in scores)[ti]
    cur448=CUR[ti].astype(float)/255.; fut448=FUT[ti].astype(float)/255.
    cols=[("future frame (GT)",fut448,None,None),
          ("planner PRED future",fut448,FP[ti],w),("planner PRED current",cur448,CP[ti],w)]
    fig,ax=plt.subplots(1,3,figsize=(9,3.2))
    for ci,(title,base,feat,ww) in enumerate(cols):
        ax[ci].imshow(base if feat is None else ov(feat,ww,base)); ax[ci].axis("off")
        ax[ci].set_title(title,fontsize=9)
    ax[0].axis("on"); ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_ylabel(f"'{w}'",fontsize=11)
    fig.suptitle(f"#{rank+1} score={sc:.2f} [{suite}]  |  {ins[:52]}",fontsize=10); fig.tight_layout()
    fig.savefig(f"{OUTDIR}/top_{rank:02d}_{suite}_{w}.png",dpi=95,bbox_inches="tight"); plt.close(fig)
    print(f"saved top_{rank:02d}_{suite}_{w}.png score={sc:.3f}",flush=True)
print("ALLDONE",flush=True)
