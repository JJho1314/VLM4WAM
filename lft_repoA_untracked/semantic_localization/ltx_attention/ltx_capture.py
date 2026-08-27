"""LTX semantic cross-attention capture: run the trained semantic-plan LTX on LIBERO clips,
capture video->SigLIP-plan attention (all 28 blocks' semantic_attn), overlay. Answers whether
the semantic-guided LTX's attention focuses on target objects."""
import os, sys, json, ast, math, av, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
GE="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/ge_act"
os.chdir(GE); sys.path.insert(0, GE)
from utils.model_utils import load_diffusion_model, load_vae_models, forward_pass
from utils.data_utils import _normalize_latents, _pack_latents
from models.ltx_models.transformer_ltx_multiview import LTXVideoTransformer3DModel, apply_rotary_emb
from models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from models.ltx_models.semantic_conditioning import OnlineSiglip2SemanticEncoder, build_semantic_plan_times, select_future_keyframes
from transformers import T5EncoderModel, T5Tokenizer
dev="cuda"; dt=torch.bfloat16
D="/data/LFT-W02_data/junjie/ltx_semantic_ckpt"; LTX="/data/LFT-W02_data/junjie/weights/LTX-Video"
SIG="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-large-patch16-256"
NPREV=4; NFUT=9; KFI=[0,3,5,8]; SS=0.5
def log(m): print(m, flush=True)

cfg=ast.literal_eval(json.load(open(f"{D}/config.json"))["diffusion_model"])["config"]
model=load_diffusion_model(LTXVideoTransformer3DModel, model_dir=f"{D}/step_25000", load_weights=True, **cfg).to(dev,dt).eval()
vae=load_vae_models(AutoencoderKLLTXVideo, f"{LTX}/vae").to(dev,dt).eval()
if isinstance(vae.latents_mean,list): vae.latents_mean=torch.tensor(vae.latents_mean)
if isinstance(vae.latents_std,list): vae.latents_std=torch.tensor(vae.latents_std)
tok=T5Tokenizer.from_pretrained(f"{LTX}/tokenizer"); te=T5EncoderModel.from_pretrained(f"{LTX}/text_encoder").to(dev,dt).eval()
sem_enc=OnlineSiglip2SemanticEncoder(SIG, device=dev, dtype=dt)
log("models loaded")

# ---- capturing semantic processor ----
CAP={}
class CapSem:
    def __init__(self, idx): self.idx=idx
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 query_rotary_emb=None, key_rotary_emb=None):
        q=attn.to_q(hidden_states); k=attn.to_k(encoder_hidden_states); v=attn.to_v(encoder_hidden_states)
        if attn.norm_q is not None: q=attn.norm_q(q)
        if attn.norm_k is not None: k=attn.norm_k(k)
        if query_rotary_emb is not None: q=apply_rotary_emb(q, query_rotary_emb)
        if key_rotary_emb is not None: k=apply_rotary_emb(k, key_rotary_emb)
        q=q.unflatten(2,(attn.heads,-1)).transpose(1,2); k=k.unflatten(2,(attn.heads,-1)).transpose(1,2); v=v.unflatten(2,(attn.heads,-1)).transpose(1,2)
        with torch.no_grad():
            scale=1.0/math.sqrt(q.shape[-1])
            probs=(q.float()@k.float().transpose(-1,-2)*scale).softmax(-1)  # [B,H,Sq,Sk]
            # softmax sums to 1 over keys, so MAX-over-plan-tokens (not mean) shows which video
            # patches lock strongly onto the semantic plan; mean over heads.
            CAP[self.idx]=probs.mean(1).amax(-1).float().cpu()  # [B,Sq]
        out=F.scaled_dot_product_attention(q,k,v,attn_mask=attention_mask,dropout_p=0.0,is_causal=False)
        out=out.transpose(1,2).flatten(2,3).to(q.dtype)
        out=attn.to_out[0](out); out=attn.to_out[1](out)
        return out
for i,blk in enumerate(model.transformer_blocks):
    if getattr(blk,"semantic_cross_attention",False): blk.semantic_attn.set_processor(CapSem(i))
log("processors installed")

# ---- read one LIBERO clip (main+wrist, 13 frames, 256) ----
ROOT="/data/LFT-W02_data/junjie/data/LIBERO-fastwam"
def read_clip(suite, ei, n=NPREV+NFUT):
    d=f"{ROOT}/libero_{suite}_no_noops_lerobot/videos/chunk-000"
    out=[]
    for cam in ("observation.images.image","observation.images.wrist_image"):
        c=av.open(f"{d}/{cam}/episode_{ei:06d}.mp4"); frs=[]
        for j,fr in enumerate(c.decode(video=0)):
            if j>=n: break
            frs.append(np.asarray(Image.fromarray(fr.to_ndarray(format="rgb24")).resize((256,256))))
        c.close(); out.append(np.stack(frs))
    return np.stack(out)  # [2, n, 256, 256, 3]
def task_of(suite, ei):
    for l in open(f"{ROOT}/libero_{suite}_no_noops_lerobot/meta/episodes.jsonl"):
        e=json.loads(l)
        if e["episode_index"]==ei: return e["tasks"][0]
    return ""

suite,ei="object",0
clip=read_clip(suite,ei); prompt=task_of(suite,ei)
log(f"clip {clip.shape} prompt: {prompt[:50]}")
vid=torch.from_numpy(clip).permute(0,1,4,2,3).float()/255.*2-1  # [2,13,3,256,256]
vid=vid.to(dev,dt)
mem=vid[:, :NPREV]; fut=vid[:, NPREV:NPREV+NFUT]  # [2,4,..],[2,9,..]

@torch.no_grad()
def enc(v):  # v [V,T,3,H,W] -> normalized packed latents (V, seq, C), and (t,h,w)
    x=v.permute(0,2,1,3,4)  # [V,3,T,H,W]
    lat=vae.encode(x).latent_dist.sample()
    lat=_normalize_latents(lat.to(dt), vae.latents_mean.to(dev), vae.latents_std.to(dev))
    t,h,w=lat.shape[2],lat.shape[3],lat.shape[4]
    return _pack_latents(lat, 1, 1), (t,h,w)
with torch.no_grad():
    # mem: each frame -> 1 latent frame (encode per-frame)
    memx=mem.reshape(-1,1,3,256,256).permute(0,2,1,3,4)  # [(V*4),3,1,256,256]
    ml=vae.encode(memx).latent_dist.sample(); ml=_normalize_latents(ml.to(dt),vae.latents_mean.to(dev),vae.latents_std.to(dev))
    _,c,_,h,w=ml.shape; ml=ml.reshape(2,NPREV,c,1,h,w).permute(0,2,1,4,5,3).reshape(2,c,NPREV,h,w)  # [2,c,4,h,w]
    fl,(ft,fh,fw)=enc(fut); fl=fl.reshape(2,ft,fh,fw,c).permute(0,4,1,2,3)  # [2,c,ft,h,w]
    latents=torch.cat((ml,fl),dim=2)  # [2,c,T=4+ft,h,w]
    T=latents.shape[2]
    latents_packed=_pack_latents(latents,1,1)  # [2, T*h*w, c]
    noise=torch.randn_like(latents_packed); noisy=(1-SS)*latents_packed+SS*noise
    # T5
    ti=tok([prompt],padding="max_length",max_length=128,truncation=True,return_tensors="pt")
    pe=te(ti.input_ids.to(dev),attention_mask=ti.attention_mask.to(dev))[0]
    pmask=ti.attention_mask.to(dev)
    # semantic plan: siglip of future keyframes [0,3,5,8]
    kf=fut[:, KFI]  # [2,4,3,256,256]
    sem=sem_enc.encode(kf.unsqueeze(0))  # [1,2,4,256,1024]
    times=build_semantic_plan_times(1,2,NPREV,NFUT,T,tuple(KFI),device=dev,dtype=torch.float32)
    log(f"latents {tuple(latents_packed.shape)} T={T} h={h} sem {tuple(sem.shape)} times {tuple(times.shape)}")
    CAP.clear()
    ts=torch.full((2,),500,device=dev,dtype=torch.long)
    forward_pass(model, pe, pmask, noisy, ts, num_frames=T, height=h, width=w, n_view=2,
                 temporal_compression_ratio=vae.temporal_compression_ratio, spatial_compression_ratio=vae.spatial_compression_ratio,
                 semantic_plan=sem, semantic_plan_times=times)
    log(f"captured {len(CAP)} blocks, map shape {tuple(CAP[list(CAP)[0]].shape)}")

# overlay: avg blocks, reshape (T,h,w), take mem frame 0 (initial obs) per view
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
SEL=[8,12,16,20]; TURBO=cm.get_cmap("turbo")
amap=torch.stack([CAP[b] for b in SEL if b in CAP]).mean(0).reshape(2,T,h,w)  # [2,T,h,w]
np.savez(f"{D}/_ltx_one_maps.npz", amap=amap.numpy(), clip=clip, prompt=prompt, T=T, h=h)
log(f"attn std across frames: {[round(float(amap[0,t].std()),4) for t in range(T)]}")
def sharp(g,lo=50,hi=99,gamma=1.4):
    g=(g-g.min())/(g.max()-g.min()+1e-6); plo,phi=np.percentile(g,[lo,hi]); return (np.clip((g-plo)/(phi-plo+1e-6),0,1))**gamma
def ov(view, tframe):
    up=F.interpolate(torch.from_numpy(sharp(amap[view,tframe].numpy()))[None,None].float(),(256,256),mode="bilinear")[0,0].numpy()
    return clip[view,0].astype(float)/255.*0.5+TURBO(up)[...,:3]*0.5
# show main+wrist across several latent frames to pick the informative one
fig,ax=plt.subplots(2,T,figsize=(2.4*T,5))
for vw in (0,1):
    for tf in range(T):
        ax[vw,tf].imshow(ov(vw,tf)); ax[vw,tf].axis("off")
        if vw==0: ax[vw,tf].set_title(f"lat-frame {tf}",fontsize=8)
    ax[vw,0].axis("on"); ax[vw,0].set_xticks([]); ax[vw,0].set_yticks([]); ax[vw,0].set_ylabel(("main","wrist")[vw],fontsize=11)
fig.suptitle(f"LTX semantic attn (max-over-plan) [{prompt[:45]}]",fontsize=11); fig.tight_layout()
fig.savefig(f"{D}/_ltx_one.png",dpi=100,bbox_inches="tight"); plt.close(fig)
log("SAVED _ltx_one.png"); log("ALLDONE")
