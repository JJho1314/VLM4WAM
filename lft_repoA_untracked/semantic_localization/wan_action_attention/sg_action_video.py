"""SG-WAM action-to-video attention (MaskWAM-style, our semantic-guided joint model):
capture the action expert's attention to video (action_blocks.attn2) WITH semantic plan (SG) vs
WITHOUT (baseline). Same model, isolating the semantic guidance effect. 2 rows: baseline / SG."""
import os, sys, json, ast, math, av, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
GE="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/ge_act"
os.chdir(GE); sys.path.insert(0, GE)
from utils.model_utils import load_diffusion_model, load_vae_models
from models.ltx_models.transformer_ltx_multiview import LTXVideoTransformer3DModel
from models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from models.ltx_models.semantic_conditioning import OnlineSiglip2SemanticEncoder, build_semantic_plan_times
from transformers import T5EncoderModel, T5Tokenizer
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
dev="cuda"; dt=torch.bfloat16
JD="/data/LFT-W02_data/junjie/weights/joint_vlm_geact_action_k4_50k/step_40000/ltx"
LTX="/data/LFT-W02_data/junjie/weights/LTX-Video"
SIG="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-large-patch16-256"
ROOT="/data/LFT-W02_data/junjie/data/LIBERO-fastwam"
TPL="A video recorded from a robot's point of view executing the following instruction: {task}"
NPREV=4; NFUT=9; KFI=[0,3,5,8]; SS=0.15; VIEW=0; LATF=4; RES=256; AH=32
def log(m): print(m, flush=True)

cfg=json.load(open(f"{JD}/config.json"))
model=load_diffusion_model(LTXVideoTransformer3DModel, model_dir=JD, load_weights=True, **cfg).to(dev,dt).eval()
vae=load_vae_models(AutoencoderKLLTXVideo, f"{LTX}/vae").to(dev,dt).eval()
if isinstance(vae.latents_mean,list): vae.latents_mean=torch.tensor(vae.latents_mean)
if isinstance(vae.latents_std,list): vae.latents_std=torch.tensor(vae.latents_std)
tok=T5Tokenizer.from_pretrained(f"{LTX}/tokenizer"); te=T5EncoderModel.from_pretrained(f"{LTX}/text_encoder").to(dev,dt).eval()
sem_enc=OnlineSiglip2SemanticEncoder(SIG, device=dev, dtype=dt)
from utils.data_utils import _normalize_latents, _pack_latents
log("models loaded")

# hook action_blocks[i].attn2 : action queries -> video features
CAP={"acc":None,"n":0}
def wrap(attn):
    orig=attn.forward
    def fwd(hidden_states, encoder_hidden_states=None, **kw):
        if encoder_hidden_states is not None:
            with torch.no_grad():
                q=attn.to_q(hidden_states); k=attn.to_k(encoder_hidden_states)
                if attn.norm_q is not None: q=attn.norm_q(q)
                if attn.norm_k is not None: k=attn.norm_k(k)
                H=attn.heads; D=q.shape[-1]//H
                q=q.unflatten(-1,(H,D)).transpose(1,2).float(); k=k.unflatten(-1,(H,D)).transpose(1,2).float()
                probs=(q@k.transpose(-1,-2)/math.sqrt(D)).softmax(-1)  # [B,H,Sa,Sv]
                m=probs.mean(1).mean(1)[0].cpu().numpy()  # mean heads+action-queries -> [Sv]
                CAP["acc"]=m if CAP["acc"] is None else CAP["acc"]+m; CAP["n"]+=1
        return orig(hidden_states, encoder_hidden_states=encoder_hidden_states, **kw)
    attn.forward=fwd
nhook=0
for blk in getattr(model,"action_blocks",[]):
    if hasattr(blk,"attn2"): wrap(blk.attn2); nhook+=1
log(f"hooked {nhook} action_blocks.attn2")

def read_clip(suite,ei,n=NPREV+NFUT):
    d=f"{ROOT}/libero_{suite}_no_noops_lerobot/videos/chunk-000"; out=[]
    for cam in ("observation.images.image","observation.images.wrist_image"):
        c=av.open(f"{d}/{cam}/episode_{ei:06d}.mp4"); frs=[]
        for j,fr in enumerate(c.decode(video=0)):
            if j>=n: break
            frs.append(np.asarray(Image.fromarray(fr.to_ndarray(format="rgb24")).resize((RES,RES))))
        c.close(); out.append(np.stack(frs))
    return np.stack(out)
def episodes(suite): return [(json.loads(l)["episode_index"],json.loads(l)["tasks"][0]) for l in open(f"{ROOT}/libero_{suite}_no_noops_lerobot/meta/episodes.jsonl")]
import pandas as pd, glob
STATS=json.load(open("/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/ge_act/configs/ltx_model/libero/libero_fastwam_mix.json"))
def _mm(x,st):  # q01/q99 min-max to [-1,1]
    lo=np.array(st["q01"]); hi=np.array(st["q99"]); return np.clip(2*(x-lo)/(hi-lo+1e-6)-1,-1,1)
def load_action_state(suite,ei):
    d=f"{ROOT}/libero_{suite}_no_noops_lerobot/data/chunk-000/episode_{ei:06d}.parquet"
    df=pd.read_parquet(d)
    act=np.stack(df["action"].values)  # [L,7]
    st=np.stack(df["observation.state"].values)  # [L,8]
    s0=NPREV-1  # current observation frame
    a=act[s0:s0+AH]  # [AH,7]
    if len(a)<AH: a=np.concatenate([a,np.repeat(a[-1:],AH-len(a),0)],0)
    key=f"libero_{suite}_no_noops_lerobot"
    a=_mm(a,STATS[key+"_delta_eef"]); s=_mm(st[s0],STATS[key+"_state_eef"])  # [AH,7],[8]
    a14=np.concatenate([a,np.zeros((AH,7))],1); s14=np.concatenate([s,np.zeros(6)])  # pad->14
    return torch.tensor(a14,device=dev,dtype=dt)[None], torch.tensor(s14,device=dev,dtype=dt)[None,None]
def caption(p):
    ti=tok([p],padding="max_length",max_length=128,truncation=True,return_tensors="pt")
    return te(ti.input_ids.to(dev),attention_mask=ti.attention_mask.to(dev))[0]

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

SCENES=[]; seen=set()
for suite in ("object","spatial","goal","10"):
    for ei,task in episodes(suite):
        if task in seen: continue
        seen.add(task); SCENES.append((suite,ei,task))
        if len([s for s in SCENES if s[0]==suite])>=2: break
SCENES=SCENES[:6]; log(f"scenes: {[s[2][:24] for s in SCENES]}")

@torch.no_grad()
def run(dic, use_sem):
    CAP["acc"]=None; CAP["n"]=0
    T,h,w=dic["T"],dic["h"],dic["w"]
    SSA=0.1  # light noise on the REAL action so queries stay task-specific
    action=(1-SSA)*dic["act"]+SSA*torch.randn_like(dic["act"])
    proprio=dic["state"]
    ts=torch.full((2,),500,device=dev,dtype=torch.long); ta=torch.full((1,AH),int(SSA*1000),device=dev,dtype=torch.long)
    times=build_semantic_plan_times(1,2,NPREV,NFUT,T,tuple(KFI),device=dev,dtype=torch.float32) if use_sem else None
    lfr=1.0/(30/8.0); rope=[lfr,32,32]
    model(hidden_states=dic["noisy"], encoder_hidden_states=dic["pe"], timestep=ts, n_view=2,
          action_states=action, action_timestep=ta, return_action=True, return_video=True, history_action_state=proprio,
          num_frames=T, height=h, width=w, rope_interpolation_scale=rope, return_dict=False,
          semantic_plan=(dic["sem"] if use_sem else None), semantic_plan_times=times)
    if CAP["acc"] is None: return None
    a=CAP["acc"]/max(CAP["n"],1)  # [Sv_all views]; take VIEW's first frame
    per=T*h*w
    return a[VIEW*per:VIEW*per+per].reshape(T,h,w)[LATF]

data=[]
for suite,ei,task in SCENES:
    clip=read_clip(suite,ei); noisy,sem,T,h,w=prep(clip); act,state=load_action_state(suite,ei)
    data.append(dict(clip=clip,noisy=noisy,sem=sem,T=T,h=h,w=w,pe=caption(task),task=task,act=act,state=state))
log("inputs prepared")

TURBO=cm.get_cmap("turbo")
def sharp(g,lo=50,hi=99.5,gamma=1.7):
    g=(g-g.min())/(g.max()-g.min()+1e-6); plo,phi=np.percentile(g,[lo,hi]); return (np.clip((g-plo)/(phi-plo+1e-6),0,1))**gamma
def ov(clip,g):
    g=g.copy().astype(float); s=np.sort(g.ravel())
    if len(s)>3: g=np.minimum(g,s[-3])  # clip top-2 sink outliers
    up=F.interpolate(torch.from_numpy(sharp(g))[None,None].float(),(RES,RES),mode="bilinear")[0,0].numpy()
    return clip[VIEW,0].astype(float)/255.*0.45+TURBO(up)[...,:3]*0.55

def nrm(g): g=g-g.min(); return g/(g.max()+1e-6)
def ov_diff(clip,d):  # d in [-1,1]; show only POSITIVE (SG attends more) via turbo
    dp=np.clip(d,0,None); dp=dp/(dp.max()+1e-6)
    up=F.interpolate(torch.from_numpy(dp)[None,None].float(),(RES,RES),mode="bilinear")[0,0].numpy()
    return clip[VIEW,0].astype(float)/255.*0.5+TURBO(up)[...,:3]*0.5
N=len(data); fig,ax=plt.subplots(3,N,figsize=(3*N,9.2))
for c in range(N):
    base=run(data[c], use_sem=False); sg=run(data[c], use_sem=True)
    ax[0,c].imshow(ov(data[c]["clip"],base) if base is not None else data[c]["clip"][VIEW,0]/255.); ax[0,c].axis("off"); ax[0,c].set_title(data[c]["task"][:24],fontsize=8)
    ax[1,c].imshow(ov(data[c]["clip"],sg) if sg is not None else data[c]["clip"][VIEW,0]/255.); ax[1,c].axis("off")
    if base is not None and sg is not None:
        diff=nrm(sg)-nrm(base)  # positive = SG attends more than baseline
        ax[2,c].imshow(ov_diff(data[c]["clip"],diff))
    ax[2,c].axis("off")
    log(f"scene {c} done")
for r,lab in ((0,"baseline (no plan)"),(1,"SG-WAM (semantic plan)"),(2,"SG - baseline (added focus)")):
    ax[r,0].axis("on"); ax[r,0].set_xticks([]); ax[r,0].set_yticks([]); ax[r,0].set_ylabel(lab,fontsize=11,fontweight="bold")
fig.suptitle("SG-WAM action-to-video attention: baseline vs semantic-guided (LIBERO, main cam)",fontsize=13)
fig.tight_layout(); fig.savefig("/data/LFT-W02_data/junjie/ltx_semantic_ckpt/sg_action_video.png",dpi=108,bbox_inches="tight"); plt.close(fig)
log("SAVED sg_action_video.png"); log("ALLDONE")
