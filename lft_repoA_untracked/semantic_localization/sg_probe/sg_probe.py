"""SG-WAM feature probe: train a lightweight loc-head to localize the target object FROM the WAM's
video features, separately on BASELINE (no plan) vs SG (semantic plan) features. If SG features give
higher target-localization (soft-IoU), the semantic guidance made the representation more target-aware.
Distills CLIPSeg pseudo-GT. Video feature = a mid video-DiT block output, 8x8 grid, 2048-d."""
import os, sys, json, re, math, av, random, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from PIL import Image
GE="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/ge_act"
os.chdir(GE); sys.path.insert(0, GE)
from utils.model_utils import load_diffusion_model, load_vae_models
from utils.data_utils import _normalize_latents, _pack_latents
from models.ltx_models.transformer_ltx_multiview import LTXVideoTransformer3DModel
from models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo
from models.ltx_models.semantic_conditioning import OnlineSiglip2SemanticEncoder, build_semantic_plan_times
from transformers import T5EncoderModel, T5Tokenizer
from transformers.models.clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
dev="cuda"; dt=torch.bfloat16; torch.manual_seed(0); random.seed(1)
JD="/data/LFT-W02_data/junjie/weights/joint_vlm_geact_action_k4_50k/step_40000/ltx"
LTX="/data/LFT-W02_data/junjie/weights/LTX-Video"
SIG="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/siglip2-large-patch16-256"
ROOT="/data/LFT-W02_data/junjie/data/LIBERO-fastwam"
TPL="A video recorded from a robot's point of view executing the following instruction: {task}"
NPREV=4; NFUT=9; KFI=[0,3,5,8]; SS=0.15; VIEW=0; LATF=5; RES=256; PROBE_BLOCK=16
NOUNS=["bowl","plate","mug","pot","cabinet","drawer","stove","basket","soup","banana","cheese","cream","ketchup","milk","butter","sauce","tomato","bottle","cup","moka","wine","book","mustard","box","pudding","microwave"]
def log(m): print(m, flush=True)

cfg=json.load(open(f"{JD}/config.json"))
model=load_diffusion_model(LTXVideoTransformer3DModel, model_dir=JD, load_weights=True, **cfg).to(dev,dt).eval()
vae=load_vae_models(AutoencoderKLLTXVideo, f"{LTX}/vae").to(dev,dt).eval()
if isinstance(vae.latents_mean,list): vae.latents_mean=torch.tensor(vae.latents_mean)
if isinstance(vae.latents_std,list): vae.latents_std=torch.tensor(vae.latents_std)
tok=T5Tokenizer.from_pretrained(f"{LTX}/tokenizer"); te=T5EncoderModel.from_pretrained(f"{LTX}/text_encoder").to(dev,dt).eval()
sem_enc=OnlineSiglip2SemanticEncoder(SIG, device=dev, dtype=dt)
cproc=CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
cseg=CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
for p in cseg.parameters(): p.requires_grad_(False)
log("models loaded")

# hook mid video block output
FEAT={"v":None}
def hook(mod,inp,out):
    FEAT["v"]=(out[0] if isinstance(out,tuple) else out).detach()
model.transformer_blocks[PROBE_BLOCK].register_forward_hook(hook)

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
def extract(dic, use_sem):
    T,h,w=dic["T"],dic["h"],dic["w"]; FEAT["v"]=None
    times=build_semantic_plan_times(1,2,NPREV,NFUT,T,tuple(KFI),device=dev,dtype=torch.float32) if use_sem else None
    lfr=1.0/(30/8.0)
    model(hidden_states=dic["noisy"], encoder_hidden_states=dic["pe"], timestep=torch.full((2,),500,device=dev,dtype=torch.long),
          n_view=2, num_frames=T, height=h, width=w, rope_interpolation_scale=[lfr,32,32], return_dict=False, return_video=True, return_action=False,
          semantic_plan=(dic["sem"] if use_sem else None), semantic_plan_times=times)
    f=FEAT["v"]  # [(b v), Sv, 2048] ; take VIEW, first latent frame -> [h*w, 2048]
    per=T*h*w
    return f[VIEW].reshape(T,h,w,-1)[LATF].reshape(h*w,-1).float()  # [64,2048]

@torch.no_grad()
def temb(w):
    inp=cproc(text=[f"a photo of a {w}"],return_tensors="pt",padding="max_length",max_length=77)
    inp={k:v.to(dev) for k,v in inp.items() if k in ("input_ids","attention_mask")}
    o=cseg.clip.text_model(**inp); t=o.pooler_output if hasattr(o,"pooler_output") else o[1]
    return F.normalize(t.float(),dim=-1)
@torch.no_grad()
def cmask(comp,w,g):
    img=Image.fromarray(comp.astype("uint8")).resize((352,352))
    inp=cproc(text=[w],images=[img],return_tensors="pt",padding=True); inp={k:v.to(dev) for k,v in inp.items()}
    lg=cseg(**inp).logits
    if lg.ndim==2: lg=lg[None]
    return F.interpolate(torch.sigmoid(lg[0])[None,None],(g,g),mode="bilinear")[0,0]

OG=32  # probe output grid (bilinear-upsampled localization map)
class Probe(nn.Module):  # LINEAR probe: logit[patch] = feature[patch] . (W @ text)  -- exposes feature quality
    def __init__(s,D=2048,tdim=512):
        super().__init__(); s.W=nn.Linear(tdim,D,bias=False); s.b=nn.Linear(tdim,1)
        s.fn=nn.LayerNorm(D,elementwise_affine=False)
    def forward(s,f,t,g):  # f [B,g*g,D], t [B,tdim]
        q=s.W(t); logit=(s.fn(f)*q[:,None,:]).sum(-1)*(f.shape[-1]**-0.5)+s.b(t)  # [B,g*g]
        x=logit.reshape(f.shape[0],1,g,g)
        return F.interpolate(x,(OG,OG),mode="bilinear",align_corners=False)[:,0]

# ---- collect scenes (multiple episodes per task for a bigger, robust sample) ----
N_EP=4  # episodes (distinct initial layouts) per task
SCENES=[]; cnt={}
for suite in ("object","spatial","goal","10"):
    for ei,task in episodes(suite):
        ws=[w for w in re.findall(r"[a-z]+",task.lower()) if w in NOUNS]
        if not ws or cnt.get(task,0)>=N_EP: continue
        cnt[task]=cnt.get(task,0)+1; SCENES.append((suite,ei,task,ws[0]))
log(f"{len(SCENES)} scenes ({len(cnt)} tasks x up to {N_EP} eps)")
DATA=[]
for suite,ei,task,noun in SCENES:
    clip=read_clip(suite,ei); noisy,sem,T,h,w=prep(clip)
    dic=dict(noisy=noisy,sem=sem,T=T,h=h,w=w,pe=caption(task))
    bf=extract(dic,False); sf=extract(dic,True)
    if len(DATA)==0:
        rel=float((bf-sf).norm()/(bf.norm()+1e-6)); log(f"  [sanity] baseline vs SG feature rel-diff = {rel:.3f} (0=plan not applied)")
    g=h  # 8 (feature grid)
    m=cmask(clip[VIEW,0],noun,32)  # CLIPSeg target mask at output grid 32
    DATA.append(dict(clip=clip,noun=noun,task=task,bf=bf,sf=sf,mask=m,g=g))
log(f"features extracted (feat grid={DATA[0]['g']}, output grid=32)")
def fit(TR,key):
    p=Probe().to(dev); opt=torch.optim.AdamW(p.parameters(),lr=1e-3)
    for step in range(600):
        d=random.choice(TR); logit=p(d[key][None].to(dev),temb(d["noun"]),d["g"])
        loss=F.binary_cross_entropy_with_logits(logit,d["mask"][None].to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    return p.eval()
@torch.no_grad()
def peakhit(p,d,key):
    pr=torch.sigmoid(p(d[key][None].to(dev),temb(d["noun"]),d["g"])[0]).cpu().numpy()
    return float(d["mask"].cpu().numpy()[np.unravel_index(pr.argmax(),pr.shape)])
# ---- K-fold CV: every scene held out once; per-scene paired (baseline,SG) peak-hit ----
K=5; idx=list(range(len(DATA))); random.shuffle(idx); folds=[idx[i::K] for i in range(K)]
PB=np.zeros(len(DATA)); PS=np.zeros(len(DATA)); probes={}
for fi,te in enumerate(folds):
    tr=[DATA[j] for j in idx if j not in te]
    pb=fit(tr,"bf"); ps=fit(tr,"sf"); probes[fi]=(pb,ps,te)
    for j in te: PB[j]=peakhit(pb,DATA[j],"bf"); PS[j]=peakhit(ps,DATA[j],"sf")
    log(f"  fold {fi+1}/{K} done")
diff=PS-PB
from scipy import stats
tt=stats.ttest_rel(PS,PB); wl=stats.wilcoxon(PS,PB) if np.any(diff!=0) else None
log(f"=== SG-WAM LINEAR feature probe ({len(DATA)} scenes, {K}-fold CV) ===")
log(f"  peak-hit  baseline={PB.mean():.3f}±{PB.std():.3f}  SG={PS.mean():.3f}±{PS.std():.3f}  d=+{diff.mean():.3f}")
log(f"  SG>base win-rate={float((diff>0).mean()):.0%};  paired t p={tt.pvalue:.2e};  Wilcoxon p={(wl.pvalue if wl else float('nan')):.2e}")
iou_b,iou_s=PB.mean(),PS.mean()
pb,ps,TE_idx=probes[0]; TE=[DATA[j] for j in TE_idx][:8]

# render test scenes: baseline-probe vs SG-probe heatmaps
TURBO=cm.get_cmap("turbo")
def up(g_): return F.interpolate(torch.from_numpy(g_)[None,None].float(),(RES,RES),mode="bilinear")[0,0].numpy()
def ov(clip,hm):
    hm=(hm-hm.min())/(hm.max()-hm.min()+1e-6); return clip[VIEW,0].astype(float)/255.*0.5+TURBO(up(hm))[...,:3]*0.5
Nn=len(TE); fig,ax=plt.subplots(2,Nn,figsize=(3*Nn,6.4))
if Nn==1: ax=ax[:,None]
for c,d in enumerate(TE):
    with torch.no_grad():
        hb=torch.sigmoid(pb(d["bf"][None].to(dev),temb(d["noun"]),d["g"])[0]).cpu().numpy()
        hs=torch.sigmoid(ps(d["sf"][None].to(dev),temb(d["noun"]),d["g"])[0]).cpu().numpy()
    ax[0,c].imshow(ov(d["clip"],hb)); ax[0,c].axis("off"); ax[0,c].set_title(f"{d['task'][:20]} [{d['noun']}]",fontsize=8)
    ax[1,c].imshow(ov(d["clip"],hs)); ax[1,c].axis("off")
ax[0,0].axis("on"); ax[0,0].set_xticks([]); ax[0,0].set_yticks([]); ax[0,0].set_ylabel(f"baseline feat (IoU {iou_b:.2f})",fontsize=10,fontweight="bold")
ax[1,0].axis("on"); ax[1,0].set_xticks([]); ax[1,0].set_yticks([]); ax[1,0].set_ylabel(f"SG feat (IoU {iou_s:.2f})",fontsize=10,fontweight="bold")
fig.suptitle("SG-WAM feature probe: target localization from baseline vs semantic-guided video features",fontsize=12)
fig.tight_layout(); fig.savefig("/data/LFT-W02_data/junjie/ltx_semantic_ckpt/sg_probe.png",dpi=108,bbox_inches="tight"); plt.close(fig)
log("SAVED sg_probe.png"); log("ALLDONE")
