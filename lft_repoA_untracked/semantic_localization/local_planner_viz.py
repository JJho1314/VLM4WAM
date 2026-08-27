"""LOCAL: load trained loc-head + planner-predicted features (dumped from pod), produce the
deliverable figure — for each noun, localize on: teacher SigLIP feat vs planner-PREDICTED future
feat vs planner-PREDICTED current feat. Shows the planner's predicted features carry object
location that a trained head can read out (the '接个头' idea)."""
import os, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
dev="cuda"
OUTDIR="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/figs"; HEADDIR="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization"
from transformers import AutoModel
from transformers.models.clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor
SIG="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-large-patch16-256"
sig=AutoModel.from_pretrained(SIG).eval().to(dev)
for p in sig.parameters(): p.requires_grad_(False)
cproc=CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
cseg=CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
for p in cseg.parameters(): p.requires_grad_(False)
print("models loaded",flush=True)

@torch.no_grad()
def penult(comp):
    x=torch.from_numpy(np.ascontiguousarray(comp)).permute(2,0,1).float().unsqueeze(0)/255.
    x=F.interpolate(x,(256,256),mode="bilinear",align_corners=False); x=((x-0.5)/0.5).to(dev)
    h=sig.vision_model.embeddings(x); L=list(sig.vision_model.encoder.layers)
    for i,l in enumerate(L):
        h=l(h,None); h=h[0] if isinstance(h,tuple) else h
        if i==len(L)-2: return h[0]
@torch.no_grad()
def temb(w):
    inp=cproc(text=[f"a photo of a {w}"],return_tensors="pt",padding="max_length",max_length=77)
    inp={k:v.to(dev) for k,v in inp.items() if k in ("input_ids","attention_mask")}
    o=cseg.clip.text_model(**inp); t=o.pooler_output if hasattr(o,"pooler_output") else o[1]
    return F.normalize(t.float(),dim=-1)

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

NPZ="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/data/planner_feats_suites.npz"
d=np.load(NPZ,allow_pickle=True)
CUR,FUT,FP,CP=d["cur"],d["fut"],d["fp"],d["cp"]
PROMPTS=[str(x) for x in d["prompts"]]; NOUNS=[str(x).split("|") for x in d["nouns"]]
SUITES=[str(x) for x in d["suites"]] if "suites" in d else ["?"]*len(CUR)
print("planner samples:",len(CUR),flush=True)

TURBO=cm.get_cmap("turbo")
def to448(a):
    t=torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1).unsqueeze(0).float()
    return F.interpolate(t,(224,448),mode="bilinear",align_corners=False)[0].permute(1,2,0).clamp(0,1).numpy()
def sharpen(hm, lo=55.0, hi=99.5, gamma=1.9):
    # spread the color range: drop the diffuse low tail, stretch, gamma-boost the peak
    plo,phi=np.percentile(hm,[lo,hi]); x=np.clip((hm-plo)/(phi-plo+1e-6),0,1); return x**gamma
def ov(feat,w,base):
    hm=torch.sigmoid(head(feat[None],temb(w))[0]).detach().cpu().numpy()
    up=to448(TURBO(sharpen(hm))[...,:3]); return base*0.45+up*0.55

# planner-predicted only (no teacher column, per request)
for ti in range(len(CUR)):
    # only the target (grasped/manipulated) object = first noun; drop reference landmarks
    cur=CUR[ti]; fut=FUT[ti]; ins=PROMPTS[ti]; words=NOUNS[ti][:1]; suite=SUITES[ti]
    cur448=cur.astype(float)/255.; fut448=fut.astype(float)/255.
    fp=torch.from_numpy(FP[ti].astype(np.float32)).to(dev)
    cp=torch.from_numpy(CP[ti].astype(np.float32)).to(dev)
    cols=[("future frame (GT)",fut448,None),
          ("planner PRED future",fut448,fp),("planner PRED current",cur448,cp)]
    nr=len(words); fig,ax=plt.subplots(nr,len(cols),figsize=(3*len(cols),3*nr))
    if nr==1: ax=ax[None,:]
    for wi,w in enumerate(words):
        for ci,(title,base,feat) in enumerate(cols):
            ax[wi,ci].imshow(base if feat is None else ov(feat,w,base)); ax[wi,ci].axis("off")
            if wi==0: ax[wi,ci].set_title(title,fontsize=9)
        ax[wi,0].axis("on"); ax[wi,0].set_xticks([]); ax[wi,0].set_yticks([]); ax[wi,0].set_ylabel(f"'{w}'",fontsize=11)
    fig.suptitle(f"[{suite}] planner semantic localization  |  {ins[:55]}",fontsize=10); fig.tight_layout()
    fig.savefig(f"{OUTDIR}/planner_{suite}_{ti}.png",dpi=95,bbox_inches="tight"); plt.close(fig)
    print(f"saved planner_{suite}_{ti}.png [{ins[:42]}] words={words}",flush=True)
print("ALLDONE",flush=True)
