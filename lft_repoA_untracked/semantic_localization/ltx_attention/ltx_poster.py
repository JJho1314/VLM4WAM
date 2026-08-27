"""LTX dual-conditioning attention: capture BOTH the text cross-attention (video->T5 instruction,
FastWAM-style fraction-on-instruction) and the semantic cross-attention (video->SigLIP goal plan),
then FUSE them (geometric mean). Where both pathways agree = the target. 3 rows: text / semantic / fused."""
import os, sys, json, ast, math, av, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
GE="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/ge_act"
os.chdir(GE); sys.path.insert(0, GE)
from utils.model_utils import load_diffusion_model, load_vae_models, forward_pass
from utils.data_utils import _normalize_latents, _pack_latents
from models.ltx_models.transformer_ltx_multiview import LTXVideoTransformer3DModel, apply_rotary_emb
from models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from models.ltx_models.semantic_conditioning import OnlineSiglip2SemanticEncoder, build_semantic_plan_times
from transformers import T5EncoderModel, T5Tokenizer
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
dev="cuda"; dt=torch.bfloat16
D="/data/LFT-W02_data/junjie/ltx_semantic_ckpt"; LTX="/data/LFT-W02_data/junjie/weights/LTX-Video"
SIG="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-large-patch16-256"
ROOT="/data/LFT-W02_data/junjie/data/LIBERO-fastwam"
NPREV=4; NFUT=9; KFI=[0,3,5,8]; SS=0.15; SEL=[8,12,16,20]; VIEW=0; LATF=4; RES=256; KFSEL=3
def log(m): print(m, flush=True)

cfg=ast.literal_eval(json.load(open(f"{D}/config.json"))["diffusion_model"])["config"]
model=load_diffusion_model(LTXVideoTransformer3DModel, model_dir=f"{D}/step_25000", load_weights=True, **cfg).to(dev,dt).eval()
vae=load_vae_models(AutoencoderKLLTXVideo, f"{LTX}/vae").to(dev,dt).eval()
if isinstance(vae.latents_mean,list): vae.latents_mean=torch.tensor(vae.latents_mean)
if isinstance(vae.latents_std,list): vae.latents_std=torch.tensor(vae.latents_std)
tok=T5Tokenizer.from_pretrained(f"{LTX}/tokenizer"); te=T5EncoderModel.from_pretrained(f"{LTX}/text_encoder").to(dev,dt).eval()
sem_enc=OnlineSiglip2SemanticEncoder(SIG, device=dev, dtype=dt)
log("models loaded")

CAP_SEM={}; CAP_TXT={}; TXT={"valid":None}
class CapSem:  # video -> SigLIP plan, per-keyframe max
    def __init__(self, idx): self.idx=idx
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query_rotary_emb=None, key_rotary_emb=None):
        q=attn.to_q(hidden_states); k=attn.to_k(encoder_hidden_states); v=attn.to_v(encoder_hidden_states)
        if attn.norm_q is not None: q=attn.norm_q(q)
        if attn.norm_k is not None: k=attn.norm_k(k)
        if query_rotary_emb is not None: q=apply_rotary_emb(q, query_rotary_emb)
        if key_rotary_emb is not None: k=apply_rotary_emb(k, key_rotary_emb)
        q=q.unflatten(2,(attn.heads,-1)).transpose(1,2); k=k.unflatten(2,(attn.heads,-1)).transpose(1,2); v=v.unflatten(2,(attn.heads,-1)).transpose(1,2)
        with torch.no_grad():
            scale=1.0/math.sqrt(q.shape[-1]); pm=(q.float()@k.float().transpose(-1,-2)*scale).softmax(-1).mean(1)  # [B,Sq,Sk]
            B,Sq,Sk=pm.shape; CAP_SEM[self.idx]=pm.reshape(B,Sq,4,Sk//4).amax(-1).float().cpu()  # [B,Sq,K]
        out=F.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask,dropout_p=0.0,is_causal=False)
        out=out.transpose(1,2).flatten(2,3).to(q.dtype); return attn.to_out[1](attn.to_out[0](out))
class CapTxt:  # video -> T5 instruction, fraction of attention on valid (non-pad) tokens (FastWAM-style)
    def __init__(self, idx): self.idx=idx
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, n_view=1, cross_view_attn=False):
        q=attn.to_q(hidden_states); k=attn.to_k(encoder_hidden_states); v=attn.to_v(encoder_hidden_states)
        q=attn.norm_q(q); k=attn.norm_k(k)
        q=rearrange(q,'(b v) l c -> b (v l) c', v=n_view)  # cross-attn fuses views on query
        q=q.unflatten(2,(attn.heads,-1)).transpose(1,2); k=k.unflatten(2,(attn.heads,-1)).transpose(1,2); v=v.unflatten(2,(attn.heads,-1)).transpose(1,2)
        with torch.no_grad():
            scale=1.0/math.sqrt(q.shape[-1]); probs=(q.float()@k.float().transpose(-1,-2)*scale).softmax(-1)  # [b,H,vl,L]
            vm=TXT["valid"].float()[:,None,None,:]
            CAP_TXT[self.idx]=((probs*vm).sum(-1)/(vm.sum(-1)+1e-6)).mean(1).float().cpu()  # [b, vl]
        out=F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False)
        out=out.transpose(1,2).flatten(2,3).to(q.dtype)
        out=rearrange(out,'b (v l) c -> (b v) l c', v=n_view)  # un-fuse views (image_rotary_emb is None)
        return attn.to_out[1](attn.to_out[0](out))
for i,blk in enumerate(model.transformer_blocks):
    if getattr(blk,"semantic_cross_attention",False): blk.semantic_attn.set_processor(CapSem(i))
    blk.attn2.set_processor(CapTxt(i))
log("processors installed")

def read_clip(suite, ei, n=NPREV+NFUT):
    d=f"{ROOT}/libero_{suite}_no_noops_lerobot/videos/chunk-000"; out=[]
    for cam in ("observation.images.image","observation.images.wrist_image"):
        c=av.open(f"{d}/{cam}/episode_{ei:06d}.mp4"); frs=[]
        for j,fr in enumerate(c.decode(video=0)):
            if j>=n: break
            frs.append(np.asarray(Image.fromarray(fr.to_ndarray(format="rgb24")).resize((RES,RES))))
        c.close(); out.append(np.stack(frs))
    return np.stack(out)
def episodes(suite):
    return [(json.loads(l)["episode_index"], json.loads(l)["tasks"][0]) for l in open(f"{ROOT}/libero_{suite}_no_noops_lerobot/meta/episodes.jsonl")]
SCENES=[]; seen=set()
for suite in ("object","spatial","goal","10"):
    for ei,task in episodes(suite):
        if task in seen: continue
        seen.add(task); SCENES.append((suite,ei,task))
        if len([s for s in SCENES if s[0]==suite])>=2: break
SCENES=SCENES[:6]; log(f"scenes: {[s[2][:28] for s in SCENES]}")

@torch.no_grad()
def prep(clip):
    vid=torch.from_numpy(clip).permute(0,1,4,2,3).float().to(dev,dt)/255.*2-1
    mem=vid[:,:NPREV]; fut=vid[:,NPREV:NPREV+NFUT]
    memx=mem.reshape(-1,1,3,RES,RES).permute(0,2,1,3,4)
    ml=_normalize_latents(vae.encode(memx).latent_dist.sample().to(dt),vae.latents_mean.to(dev),vae.latents_std.to(dev))
    _,c,_,h,w=ml.shape; ml=ml.reshape(2,NPREV,c,1,h,w).permute(0,2,1,4,5,3).reshape(2,c,NPREV,h,w)
    fl=_normalize_latents(vae.encode(fut.permute(0,2,1,3,4)).latent_dist.sample().to(dt),vae.latents_mean.to(dev),vae.latents_std.to(dev))
    latents=torch.cat((ml,fl),dim=2); T=latents.shape[2]
    lp=_pack_latents(latents,1,1); noisy=(1-SS)*lp+SS*torch.randn_like(lp)
    sem=sem_enc.encode(fut[:,KFI].unsqueeze(0))
    return noisy,sem,T,h,w
@torch.no_grad()
def caption(p):
    ti=tok([p],padding="max_length",max_length=128,truncation=True,return_tensors="pt")
    return te(ti.input_ids.to(dev),attention_mask=ti.attention_mask.to(dev))[0], ti.attention_mask.to(dev)

data=[]
for suite,ei,task in SCENES:
    clip=read_clip(suite,ei); noisy,sem,T,h,w=prep(clip); pe,pm=caption(task)
    data.append(dict(clip=clip,noisy=noisy,sem=sem,T=T,h=h,w=w,pe=pe,pm=pm,task=task))
log("inputs prepared")

@torch.no_grad()
def run(dic):
    CAP_SEM.clear(); CAP_TXT.clear(); TXT["valid"]=dic["pm"]
    ts=torch.full((2,),500,device=dev,dtype=torch.long)
    times=build_semantic_plan_times(1,2,NPREV,NFUT,dic["T"],tuple(KFI),device=dev,dtype=torch.float32)
    forward_pass(model,dic["pe"],dic["pm"],dic["noisy"],ts,num_frames=dic["T"],height=dic["h"],width=dic["w"],n_view=2,
                 temporal_compression_ratio=vae.temporal_compression_ratio,spatial_compression_ratio=vae.spatial_compression_ratio,
                 semantic_plan=dic["sem"],semantic_plan_times=times)
    sem=torch.stack([CAP_SEM[b] for b in SEL if b in CAP_SEM]).mean(0)[...,KFSEL].reshape(2,dic["T"],dic["h"],dic["w"])[VIEW,LATF].numpy()
    txt=torch.stack([CAP_TXT[b] for b in SEL if b in CAP_TXT]).mean(0).reshape(2,dic["T"],dic["h"],dic["w"])[VIEW,LATF].numpy()
    return txt, sem

def nrm(g): g=g-g.min(); return g/(g.max()+1e-6)
TURBO=cm.get_cmap("turbo")
def sharp(g,lo=55,hi=99.5,gamma=1.9):
    g=nrm(g); plo,phi=np.percentile(g,[lo,hi]); return (np.clip((g-plo)/(phi-plo+1e-6),0,1))**gamma
def ov(clip,g):
    up=F.interpolate(torch.from_numpy(sharp(g))[None,None].float(),(RES,RES),mode="bilinear")[0,0].numpy()
    return clip[VIEW,0].astype(float)/255.*0.45+TURBO(up)[...,:3]*0.55

N=len(data); ROWS=["text attn (instruction)","semantic attn (goal plan)","FUSED (text x semantic)"]
fig,ax=plt.subplots(3,N,figsize=(3*N,9))
for c in range(N):
    txt,sem=run(data[c])
    fused=np.sqrt(nrm(txt)*nrm(sem))  # geometric mean = where both pathways agree
    for r,g in enumerate((txt,sem,fused)):
        ax[r,c].imshow(ov(data[c]["clip"],g)); ax[r,c].axis("off")
        if r==0: ax[r,c].set_title(data[c]["task"][:26],fontsize=8)
    log(f"scene {c} done")
for r in range(3):
    ax[r,0].axis("on"); ax[r,0].set_xticks([]); ax[r,0].set_yticks([]); ax[r,0].set_ylabel(ROWS[r],fontsize=11,fontweight="bold")
fig.suptitle("Semantic-guided LTX: text vs semantic vs FUSED cross-attention (LIBERO, main cam)",fontsize=13)
fig.tight_layout(); fig.savefig(f"{D}/ltx_semantic_attn.png",dpi=105,bbox_inches="tight"); plt.close(fig)
log("SAVED ltx_semantic_attn.png"); log("ALLDONE")
