"""SG-WAM prediction quality: does semantic guidance improve FUTURE-frame prediction? For each scene,
noise the future latents, predict WITH semantic plan (SG) vs WITHOUT (baseline), measure flow-matching
recon error (MSE of x0-estimate on the FUTURE latent frames) vs GT. Paired stats across scenes.
This tests the paper's core claim (Semantic-Guided *Prediction*)."""
import os, sys, json, re, av, random, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
GE="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/ge_act"
os.chdir(GE); sys.path.insert(0, GE)
from utils.model_utils import load_diffusion_model, load_vae_models
from utils.data_utils import _normalize_latents, _pack_latents
from models.ltx_models.transformer_ltx_multiview import LTXVideoTransformer3DModel
from models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from models.ltx_models.semantic_conditioning import OnlineSiglip2SemanticEncoder, build_semantic_plan_times
from transformers import T5EncoderModel, T5Tokenizer
from scipy import stats
dev="cuda"; dt=torch.bfloat16; torch.manual_seed(0); random.seed(1)
JD="/data/LFT-W02_data/junjie/weights/joint_vlm_geact_action_k4_50k/step_40000/ltx"
LTX="/data/LFT-W02_data/junjie/weights/LTX-Video"
SIG="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-large-patch16-256"
ROOT="/data/LFT-W02_data/junjie/data/LIBERO-fastwam"
NPREV=4; NFUT=9; KFI=[0,3,5,8]; RES=256; N_EP=4; SS_LIST=[0.3,0.5,0.7]
def log(m): print(m, flush=True)

cfg=json.load(open(f"{JD}/config.json"))
model=load_diffusion_model(LTXVideoTransformer3DModel, model_dir=JD, load_weights=True, **cfg).to(dev,dt).eval()
vae=load_vae_models(AutoencoderKLLTXVideo, f"{LTX}/vae").to(dev,dt).eval()
if isinstance(vae.latents_mean,list): vae.latents_mean=torch.tensor(vae.latents_mean)
if isinstance(vae.latents_std,list): vae.latents_std=torch.tensor(vae.latents_std)
tok=T5Tokenizer.from_pretrained(f"{LTX}/tokenizer"); te=T5EncoderModel.from_pretrained(f"{LTX}/text_encoder").to(dev,dt).eval()
sem_enc=OnlineSiglip2SemanticEncoder(SIG, device=dev, dtype=dt)
log("models loaded")

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
def caption(p):
    ti=tok([p],padding="max_length",max_length=128,truncation=True,return_tensors="pt")
    return te(ti.input_ids.to(dev),attention_mask=ti.attention_mask.to(dev))[0]

@torch.no_grad()
def prep(clip):  # return clean packed latent x0, T,h,w, sem
    vid=torch.from_numpy(clip).permute(0,1,4,2,3).float().to(dev,dt)/255.*2-1
    mem=vid[:,:NPREV]; fut=vid[:,NPREV:NPREV+NFUT]
    memx=mem.reshape(-1,1,3,RES,RES).permute(0,2,1,3,4)
    ml=_normalize_latents(vae.encode(memx).latent_dist.sample().to(dt),vae.latents_mean.to(dev),vae.latents_std.to(dev))
    _,c,_,h,w=ml.shape; ml=ml.reshape(2,NPREV,c,1,h,w).permute(0,2,1,4,5,3).reshape(2,c,NPREV,h,w)
    fl=_normalize_latents(vae.encode(fut.permute(0,2,1,3,4)).latent_dist.sample().to(dt),vae.latents_mean.to(dev),vae.latents_std.to(dev))
    latents=torch.cat((ml,fl),dim=2); T=latents.shape[2]
    x0=_pack_latents(latents,1,1)  # [B, T*h*w, C]
    sem=sem_enc.encode(fut[:,KFI].unsqueeze(0))
    return x0,T,h,w,sem

@torch.no_grad()
def pred_err(dic, ss, noise, use_sem):
    T,h,w=dic["T"],dic["h"],dic["w"]; x0=dic["x0"]
    noisy=(1-ss)*x0+ss*noise
    times=build_semantic_plan_times(1,2,NPREV,NFUT,T,tuple(KFI),device=dev,dtype=torch.float32) if use_sem else None
    lfr=1.0/(30/8.0)
    out=model(hidden_states=noisy, encoder_hidden_states=dic["pe"], timestep=torch.full((2,),int(ss*1000),device=dev,dtype=torch.long),
        n_view=2, num_frames=T, height=h, width=w, rope_interpolation_scale=[lfr,32,32], return_dict=False, return_video=True, return_action=False,
        semantic_plan=(dic["sem"] if use_sem else None), semantic_plan_times=times)[0]
    pv=out["video"] if isinstance(out,dict) else out  # velocity prediction, packed [B, S, C]
    x0hat=noisy-ss*pv  # flow-matching x0 estimate (noisy=(1-t)x0+t*noise, v=noise-x0)
    per=h*w; fut_sl=slice(NPREV*per, T*per)  # FUTURE latent frames only
    e=(x0hat[:,fut_sl].float()-x0[:,fut_sl].float()).pow(2).mean()
    return float(e)

# scenes
SCENES=[]; cnt={}
for suite in ("object","spatial","goal","10"):
    for ei,task in episodes(suite):
        if cnt.get(task,0)>=N_EP: continue
        cnt[task]=cnt.get(task,0)+1; SCENES.append((suite,ei,task))
log(f"{len(SCENES)} scenes")
CKF="/data/LFT-W02_data/junjie/ltx_semantic_ckpt/_sgpred_prog.npz"
EB=[]; ES=[]; start=0
if os.path.exists(CKF):
    z=np.load(CKF); EB=list(z["eb"]); ES=list(z["es"]); start=len(EB); log(f"resume from {start}")
for i in range(start,len(SCENES)):
    suite,ei,task=SCENES[i]
    clip=read_clip(suite,ei); x0,T,h,w,sem=prep(clip); pe=caption(task)
    d=dict(x0=x0,T=T,h=h,w=w,sem=sem,pe=pe)
    g=torch.Generator(device=dev).manual_seed(i)
    eb=[]; es=[]
    for ss in SS_LIST:
        noise=torch.randn(x0.shape,generator=g,device=dev,dtype=dt)
        eb.append(pred_err(d,ss,noise,False)); es.append(pred_err(d,ss,noise,True))
    EB.append(np.mean(eb)); ES.append(np.mean(es))
    del x0,sem,pe,d,noise; torch.cuda.empty_cache()
    if i%5==0: np.savez(CKF,eb=np.array(EB),es=np.array(ES))
    if i%20==0: log(f"  {i}/{len(SCENES)}")
np.savez(CKF,eb=np.array(EB),es=np.array(ES))
EB=np.array(EB); ES=np.array(ES); dff=EB-ES  # positive = SG lower error = better
tt=stats.ttest_rel(EB,ES); wl=stats.wilcoxon(EB,ES) if np.any(dff!=0) else None
log(f"=== SG-WAM future-prediction error (flow-matching MSE, {len(DATA)} scenes, noise {SS_LIST}) ===")
log(f"  baseline={EB.mean():.4f}±{EB.std():.4f}   SG={ES.mean():.4f}±{ES.std():.4f}   Δ(base-SG)=+{dff.mean():.4f}")
log(f"  SG-better win-rate={float((dff>0).mean()):.0%};  paired t p={tt.pvalue:.2e};  Wilcoxon p={(wl.pvalue if wl else float('nan')):.2e}")
log("ALLDONE")
