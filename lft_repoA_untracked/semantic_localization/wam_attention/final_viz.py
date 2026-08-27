"""FastWAM RGB-only vs SG-WAM cross-attention comparison (local, A6000).
Two models share the Qwen text pathway (text_proj from the SG ckpt); the video DiT is
base-posttrained (RGB-only) vs SG-finetuned. For each LIBERO frame we capture video->text
cross-attention (mean over valid text tokens, blocks 8/12/16/20) and overlay it. Poster: top
row RGB-only, bottom row SG-WAM."""
import os, sys, functools, hashlib, torch, numpy as np
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
torch.load = functools.partial(torch.load, weights_only=False)
FW="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/FastWAM"
COSMOS="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/cosmos-predict2.5"
for p in (COSMOS, FW+"/src", FW+"/experiments/libero"):
    if p not in sys.path: sys.path.insert(0,p)
from fastwam.models.cosmos.runtime import create_fastwam_cosmos
W="/data/LFT-W02_data/junjie/weights/Cosmos-Predict2.5-2B"
CKPT="/data/LFT-W02_data/junjie/fastwam_sg_ckpt/checkpoints/weights/step_015000.pt"
QCACHE="/data/LFT-W02_data/junjie/fastwam_sg_ckpt/libero_qwen"
OUT="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/figs"
TPL="A video recorded from a robot's point of view executing the following instruction: {task}"
dev="cuda"; dt=torch.bfloat16; SEL_BLOCKS=[8,12,16,20]; TF=5
os.makedirs(OUT, exist_ok=True)

def build(load_sg):
    m=create_fastwam_cosmos(video_dit_pretrained_path=W+"/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
        vae={"vae_pth":W+"/tokenizer.pth"},action_dim=7,proprio_dim=8,crossattn_dim=1024,coupling="mot",
        feature_layer=-1,action_hidden_dim=1024,action_ffn_dim=4096,action_attention_head_dim=128,model_dtype=dt,device=dev)
    if load_sg:
        m.load_checkpoint(CKPT)
    else:  # baseline: keep base video DiT, but copy the SG-trained text_proj so the text pathway matches
        ck=torch.load(CKPT,map_location="cpu",weights_only=False)
        m.text_proj.load_state_dict({k:v for k,v in ck["text_proj"].items()})
    return m.eval()

CAP={}
def install_capture(m, textmask_holder):
    net=m.dit["video"].net
    def wrap(idx, cross):
        orig=cross.forward
        def fwd(x, context=None, rope_emb=None, **kw):
            if context is not None and idx in SEL_BLOCKS:
                with torch.no_grad():
                    q,k,_=cross.compute_qkv(x, context, rope_emb=rope_emb)
                    scale=1.0/(q.shape[-1]**0.5)
                    scores=torch.einsum("bshd,blhd->bhsl", q.float(), k.float())*scale
                    probs=scores.softmax(dim=-1)
                    mm=textmask_holder["m"].float()[:,None,None,:]
                    amap=(probs*mm).sum(-1)/(mm.sum(-1)+1e-6)  # [B,H,S]
                    CAP[idx]=amap.mean(1).float().cpu()  # [B,S]
            return orig(x, context, rope_emb=rope_emb, **kw)
        cross.forward=fwd
    for i,blk in enumerate(net.blocks):
        if hasattr(blk,"cross_attn"): wrap(i, blk.cross_attn)

@torch.no_grad()
def vae_latents(m, comp):  # comp [224,448] uint8 -> latents [1,16,T',28,56]
    x=torch.from_numpy(np.ascontiguousarray(comp)).permute(2,0,1).float()/255.  # [3,224,448]
    x=(x*2-1)[None,:,None].repeat(1,1,TF,1,1).to(dev,dt)  # [1,3,TF,224,448] static clip
    return m._vae_encode(x).to(dt)

def load_ctx(prompt):
    h=hashlib.sha256(TPL.format(task=prompt).encode()).hexdigest()+".t5_len128.wan22ti2v5b.pt"
    d=torch.load(os.path.join(QCACHE,h),map_location="cpu",weights_only=False)
    return d["context"].to(dev,dt)[None], d["mask"].to(dev)[None]  # [1,128,3584],[1,128]

@torch.no_grad()
def attn_map(m, comp, prompt, tmh):
    lat=vae_latents(m, comp); ctx,mask=load_ctx(prompt); tmh["m"]=mask
    crossattn=m.text_proj(ctx)
    Tl=lat.shape[2]; t=torch.full((1,Tl),500.,device=dev,dtype=dt)
    CAP.clear()
    m.dit["video"].forward_standalone(lat, t, crossattn)
    maps=torch.stack([CAP[b] for b in SEL_BLOCKS]).mean(0)[0]  # [S]
    S=maps.shape[0]; hw=S//Tl; import math
    Wl=56//2; Hl=28//2  # patchified 14x28
    grid=maps.reshape(Tl,Hl,Wl)[0]  # first latent frame [14,28]
    return grid.numpy()

# ---- samples: pick diverse noun-bearing distinct prompts ----
d=np.load("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/data/libsamples.npz",allow_pickle=True)
CUR=d["cur"]; PROMPTS=[str(x) for x in d["prompts"]]
seen=set(); picks=[]
import random; random.seed(3)
order=list(range(len(CUR))); random.shuffle(order)
for i in order:
    k=PROMPTS[i][:40]
    if k in seen: continue
    seen.add(k); picks.append(i)
    if len(picks)>=6: break
print("picks:",[PROMPTS[i][:40] for i in picks],flush=True)

TURBO=cm.get_cmap("turbo")
def overlay(comp, grid):
    g=torch.from_numpy(grid)[None,None].float()
    up=F.interpolate(g,(224,448),mode="bilinear",align_corners=False)[0,0].numpy()
    up=(up-up.min())/(up.max()-up.min()+1e-6)
    base=comp.astype(float)/255.
    hm=TURBO(up)[...,:3]
    return base*0.5+hm*0.5

# process each model, store maps
results={}
for tag,load_sg in (("RGB-only",False),("SG-WAM",True)):
    print(f"building {tag}...",flush=True)
    m=build(load_sg); tmh={"m":None}; install_capture(m, tmh)
    results[tag]=[attn_map(m, CUR[i], PROMPTS[i], tmh) for i in picks]
    del m; torch.cuda.empty_cache()
    print(f"{tag} done",flush=True)
# persist maps + frames so rendering never recomputes the models
np.savez("/data/LFT-W02_data/junjie/fastwam_sg_ckpt/wam_maps.npz",
    rgb=np.stack(results["RGB-only"]), sg=np.stack(results["SG-WAM"]),
    frames=np.stack([CUR[i] for i in picks]),
    prompts=np.array([PROMPTS[i] for i in picks],dtype=object))
print("maps saved",flush=True)

# ---- poster: 2 rows x 6 cols ----
fig,ax=plt.subplots(2,6,figsize=(20,7))
for r,tag in enumerate(("RGB-only","SG-WAM")):
    for c,i in enumerate(picks):
        ax[r,c].imshow(overlay(CUR[i], results[tag][c])); ax[r,c].axis("off")
        if r==0: ax[r,c].set_title(PROMPTS[i][:32],fontsize=7)
    ax[r,0].axis("on"); ax[r,0].set_xticks([]); ax[r,0].set_yticks([])
    ax[r,0].set_ylabel(("RGB-only WAM","SG-WAM (ours)")[r],fontsize=13,fontweight="bold")
fig.suptitle("FastWAM cross-attention: RGB-only vs Semantic-Guided (LIBERO)",fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/wam_rgbonly_vs_sg.png",dpi=110,bbox_inches="tight"); plt.close(fig)
print("SAVED wam_rgbonly_vs_sg.png",flush=True); print("ALLDONE",flush=True)
