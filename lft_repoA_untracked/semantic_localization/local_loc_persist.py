"""LOCAL (persistent path): train loc-head on dumped LIBERO samples (SigLIP2 penult +
CLIPSeg distill), output teacher-feature localization figures. Robust fast-tokenizer load."""
import os, re, random, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
dev = "cuda"; torch.manual_seed(0); random.seed(11)
NPZ = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/data/libsamples.npz"
OUTDIR = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/figs"; os.makedirs(OUTDIR, exist_ok=True)
HEADDIR = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization"

from transformers import AutoModel, AutoProcessor
from transformers.models.clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor
SIG = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-large-patch16-256"
sig = AutoModel.from_pretrained(SIG).eval().to(dev)
for p in sig.parameters(): p.requires_grad_(False)
# NOTE: siglip dir has no tokenizer; use CLIPSeg's CLIP text tower for the noun conditioning
# embedding (BPE, no sentencepiece). Vision features localized are still siglip penult.
cproc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
cseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
for p in cseg.parameters(): p.requires_grad_(False)
print("models loaded", flush=True)

d = np.load(NPZ, allow_pickle=True)
CUR, PROMPTS = d["cur"], [str(x) for x in d["prompts"]]
print("samples:", len(CUR), flush=True)
NOUNS = ["bowl","plate","mug","pot","cabinet","drawer","stove","basket","soup","banana","cheese","cream","ketchup","milk","butter"]

@torch.no_grad()
def penult(comp):
    x = torch.from_numpy(np.ascontiguousarray(comp)).permute(2,0,1).float().unsqueeze(0)/255.
    x = F.interpolate(x,(256,256),mode="bilinear",align_corners=False); x=((x-0.5)/0.5).to(dev)
    h = sig.vision_model.embeddings(x); L=list(sig.vision_model.encoder.layers)
    for i,l in enumerate(L):
        h=l(h,None); h=h[0] if isinstance(h,tuple) else h
        if i==len(L)-2: return h[0]
@torch.no_grad()
def temb(w):
    inp = cproc(text=[f"a photo of a {w}"], return_tensors="pt", padding="max_length", max_length=77)
    inp = {k: v.to(dev) for k, v in inp.items() if k in ("input_ids","attention_mask")}
    o = cseg.clip.text_model(**inp)
    t = o.pooler_output if hasattr(o, "pooler_output") else o[1]
    return F.normalize(t.float(), dim=-1)  # [1,512]
@torch.no_grad()
def cgt(comp,w):
    img=Image.fromarray(comp.astype("uint8")).resize((352,352))
    inp=cproc(text=[w],images=[img],return_tensors="pt",padding=True); inp={k:v.to(dev) for k,v in inp.items()}
    lg=cseg(**inp).logits
    if lg.ndim==2: lg=lg[None]
    return F.interpolate(torch.sigmoid(lg[0])[None,None],(16,16),mode="bilinear")[0,0]

class LocHead(nn.Module):
    def __init__(self,d=1024,hid=256,tdim=512):
        super().__init__(); self.film=nn.Linear(tdim,2*hid); self.inp=nn.Conv2d(d,hid,1)
        self.net=nn.Sequential(nn.Conv2d(hid,hid,3,padding=1),nn.GroupNorm(8,hid),nn.GELU(),nn.Conv2d(hid,hid,3,padding=1),nn.GroupNorm(8,hid),nn.GELU())
        self.out=nn.Conv2d(hid,1,1)
    def forward(self,f,t):
        B=f.shape[0]; x=f.permute(0,2,1).reshape(B,-1,16,16); x=self.inp(x)
        g,b=self.film(t).chunk(2,-1); x=x*(1+g[...,None,None])+b[...,None,None]
        return self.out(self.net(x)).squeeze(1)

idx_noun=[]
for i,pr in enumerate(PROMPTS):
    ws=[w for w in re.findall(r"[a-z]+",pr.lower()) if w in NOUNS]
    if ws: idx_noun.append((i,ws))
print("noun-bearing samples:", len(idx_noun), flush=True)

head=LocHead().to(dev); opt=torch.optim.AdamW(head.parameters(),lr=3e-4)
def batch(bs=16):
    fs,ts,gs=[],[],[]
    while len(fs)<bs:
        i,ws=random.choice(idx_noun); n=random.choice(ws); comp=CUR[i]
        fs.append(penult(comp)); ts.append(temb(n)[0]); gs.append(cgt(comp,n))
    return torch.stack(fs),torch.stack(ts),torch.stack(gs)
for step in range(301):
    f,t,g=batch(16); logit=head(f,t); loss=F.binary_cross_entropy_with_logits(logit,g)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%50==0:
        with torch.no_grad():
            pr=torch.sigmoid(logit); iou=((pr>0.4)&(g>0.4)).float().sum()/(((pr>0.4)|(g>0.4)).float().sum()+1e-6)
        print(f"step {step}: loss={loss.item():.4f} soft-iou={iou.item():.3f}", flush=True)
torch.save({"state":head.state_dict()}, f"{HEADDIR}/loc_head.pt"); head.eval()
print("loc-head trained + saved", flush=True)

TURBO=cm.get_cmap("turbo")
def to448(a):
    t=torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1).unsqueeze(0).float()
    return F.interpolate(t,(224,448),mode="bilinear",align_corners=False)[0].permute(1,2,0).clamp(0,1).numpy()
def ov(feat,w,base):
    hm=torch.sigmoid(head(feat[None],temb(w))[0]).detach().cpu().numpy()
    up=to448(TURBO((hm-hm.min())/(hm.max()-hm.min()+1e-6))[...,:3]); return base*0.45+up*0.55

seen=set(); pick=[]; random.shuffle(idx_noun)
for i,ws in idx_noun:
    k=PROMPTS[i][:40]
    if k in seen: continue
    seen.add(k); pick.append((i,ws[:3]))
    if len(pick)>=5: break
for ti,(i,words) in enumerate(pick):
    comp=CUR[i]; base=comp.astype(float)/255.; feat=penult(comp)
    fig,ax=plt.subplots(1,len(words)+1,figsize=(3*(len(words)+1),3.2))
    ax[0].imshow(base); ax[0].set_title("input(2cam composite)",fontsize=9); ax[0].axis("off")
    for wi,w in enumerate(words):
        ax[wi+1].imshow(ov(feat,w,base)); ax[wi+1].axis("off"); ax[wi+1].set_title(f"'{w}'",fontsize=10)
    fig.suptitle(f"teacher-feat localization [{PROMPTS[i][:60]}]",fontsize=10); fig.tight_layout()
    fig.savefig(f"{OUTDIR}/teacher_loc_{ti}.png",dpi=95,bbox_inches="tight"); plt.close(fig)
    print(f"saved teacher_loc_{ti}.png words={words}", flush=True)
print("ALLDONE", flush=True)
