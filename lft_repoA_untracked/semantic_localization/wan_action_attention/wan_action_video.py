"""Wan FastWAM action-to-video attention (MaskWAM-style): capture the action expert's attention
to the first-frame video tokens (q_action @ k_video in the MoT mixed attention), overlay on the
observation frame. This is the attention MaskWAM visualizes."""
import os, sys, functools, hashlib, math, av, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
torch.load = functools.partial(torch.load, weights_only=False)
os.environ.setdefault("DIFFSYNTH_MODEL_ROOT", "/data/LFT-W02_data/junjie/weights")
FW="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/FastWAM"; sys.path.insert(0, FW+"/src")
from fastwam.runtime import create_fastwam
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
dt=torch.bfloat16; dev="cuda"
CKPT="/data/LFT-W02_data/junjie/eval_runs/wanfastwam_fastwam_official_v3par3/checkpoints/pytorch_model.pt"
TXT="/data/LFT-W02_data/junjie/FastWAM/data/text_embeds_cache/libero"
ROOT="/data/LFT-W02_data/junjie/data/LIBERO-fastwam"
TPL="A video recorded from a robot's point of view executing the following instruction: {task}"
NF=9; AH=32; RES=224
def log(m): print(m, flush=True)

DITCFG=dict(patch_size=[1,2,2],in_dim=48,hidden_dim=3072,ffn_dim=14336,freq_dim=256,text_dim=4096,out_dim=48,num_heads=24,attn_head_dim=128,num_layers=30,eps=1.0e-06,has_image_input=False,seperated_timestep=True,require_clip_embedding=False,require_vae_embedding=False,fuse_vae_embedding_in_latents=True,video_attention_mask_mode="first_frame_causal")
ADITCFG=dict(action_dim=7,hidden_dim=1024,ffn_dim=4096,freq_dim=256,text_dim=4096,num_heads=24,attn_head_dim=128,num_layers=30,eps=1.0e-06)
SCH={"train_shift":5.0,"infer_shift":5.0,"num_train_timesteps":1000}
model=create_fastwam("Wan-AI/Wan2.2-TI2V-5B","Wan-AI/Wan2.1-T2V-1.3B",DITCFG,load_text_encoder=False,proprio_dim=8,
    action_dit_config=ADITCFG,action_dit_pretrained_path=None,skip_dit_load_from_pretrain=True,
    video_scheduler=SCH,action_scheduler=SCH,model_dtype=dt,device=dev)
ck=torch.load(CKPT,map_location="cpu",weights_only=False)
sd={(k[len("mixtures."):] if k.startswith("mixtures.") else k):v for k,v in ck["mot"].items()}
for name,mod in model.named_modules():
    if hasattr(mod,"mixtures"): mod.mixtures.load_state_dict(sd,strict=False); MOT=mod; break
if ck.get("proprio_encoder") is not None and getattr(model,"proprio_encoder",None) is not None:
    model.proprio_encoder.load_state_dict(ck["proprio_encoder"],strict=False)
model=model.eval()
log("model loaded")

# ---- capture action->video via MoT _mixed_attention ----
CAP={"acc":None,"n":0,"sv":None}
orig_mix=MOT._mixed_attention.__func__
def mixed_cap(self,q_cat,k_cat,v_cat,attention_mask):
    Sq=q_cat.shape[1]; Sk=k_cat.shape[1]
    if Sq<Sk:  # action queries attend to [video(Sv) ; action(Sq)]
        with torch.no_grad():
            Sv=Sk-Sq; H=self.num_heads; D=q_cat.shape[-1]//H
            q=q_cat.reshape(q_cat.shape[0],Sq,H,D).permute(0,2,1,3).float()
            k=k_cat.reshape(k_cat.shape[0],Sk,H,D).permute(0,2,1,3).float()
            sc=1.0/math.sqrt(D); probs=(q@k.transpose(-1,-2)*sc).softmax(-1)  # softmax over ALL keys
            vid=probs[...,:Sv].mean(1).mean(1)[0].float().cpu().numpy()  # video part, mean heads+action-queries -> [Sv]
            CAP["acc"]=vid if CAP["acc"] is None else CAP["acc"]+vid; CAP["n"]+=1; CAP["sv"]=Sv
    return orig_mix(self,q_cat,k_cat,v_cat,attention_mask)
MOT._mixed_attention=mixed_cap.__get__(MOT,type(MOT))

def read_clip(suite,ei,n=NF):
    d=f"{ROOT}/libero_{suite}_no_noops_lerobot/videos/chunk-000/observation.images.image"
    c=av.open(f"{d}/episode_{ei:06d}.mp4"); frs=[]
    for j,fr in enumerate(c.decode(video=0)):
        if j>=n: break
        frs.append(np.asarray(Image.fromarray(fr.to_ndarray(format="rgb24")).resize((RES,RES))))
    c.close(); return np.stack(frs)  # [n,H,W,3]
def load_ctx(task):
    h=hashlib.sha256(TPL.format(task=task).encode()).hexdigest()+".t5_len128.wan22ti2v5b.pt"
    d=torch.load(os.path.join(TXT,h),map_location="cpu",weights_only=False)
    return d["context"].to(dev,dt)[None], d["mask"].to(dev)[None]
import json
def episodes(suite): return [(json.loads(l)["episode_index"],json.loads(l)["tasks"][0]) for l in open(f"{ROOT}/libero_{suite}_no_noops_lerobot/meta/episodes.jsonl")]

# pick 6 distinct-task scenes
SCENES=[]; seen=set()
for suite in ("object","spatial","goal","10"):
    for ei,task in episodes(suite):
        if task in seen: continue
        seen.add(task); SCENES.append((suite,ei,task))
        if len([s for s in SCENES if s[0]==suite])>=2: break
SCENES=SCENES[:6]; log(f"scenes: {[s[2][:26] for s in SCENES]}")

TURBO=cm.get_cmap("turbo")
def sharp(g,lo=50,hi=99.5,gamma=1.7):
    g=(g-g.min())/(g.max()-g.min()+1e-6); plo,phi=np.percentile(g,[lo,hi]); return (np.clip((g-plo)/(phi-plo+1e-6),0,1))**gamma
def overlay(frame,gmap):
    n=int(round(math.sqrt(len(gmap)))); g=gmap[:n*n].reshape(n,n)
    up=F.interpolate(torch.from_numpy(sharp(g))[None,None].float(),(RES,RES),mode="bilinear")[0,0].numpy()
    return frame.astype(float)/255.*0.45+TURBO(up)[...,:3]*0.55

@torch.no_grad()
def run(suite,ei,task):
    clip=read_clip(suite,ei)  # [9,H,W,3]
    video=torch.from_numpy(clip).permute(3,0,1,2)[None].to(dev,dt)/255.*2-1  # [1,3,9,H,W]
    img=torch.from_numpy(clip[0]).permute(2,0,1)[None].to(dev,dt)/255.*2-1  # [1,3,H,W] first frame
    ctx,cmask=load_ctx(task)
    proprio=torch.zeros(1,8,device=dev,dtype=dt)
    CAP["acc"]=None; CAP["n"]=0
    model.infer(prompt=None, input_image=img, num_frames=NF, action_horizon=AH, proprio=proprio,
                context=ctx, context_mask=cmask, num_inference_steps=4, seed=0)
    m=CAP["acc"]/max(CAP["n"],1) if CAP["acc"] is not None else None
    return clip[0], m, CAP["sv"]

N=len(SCENES); fig,ax=plt.subplots(1,N,figsize=(3*N,3.4))
if N==1: ax=[ax]
for c,(suite,ei,task) in enumerate(SCENES):
    frame,gmap,sv=run(suite,ei,task)
    log(f"scene {c}: sv={sv} map={None if gmap is None else gmap.shape}")
    ax[c].imshow(overlay(frame,gmap) if gmap is not None else frame.astype(float)/255.); ax[c].axis("off"); ax[c].set_title(task[:24],fontsize=8)
fig.suptitle("Wan FastWAM: action-to-video attention (MaskWAM-style, LIBERO)",fontsize=13)
fig.tight_layout(); fig.savefig("/data/LFT-W02_data/junjie/ltx_semantic_ckpt/wan_action_video.png",dpi=110,bbox_inches="tight"); plt.close(fig)
log("SAVED wan_action_video.png"); log("ALLDONE")
