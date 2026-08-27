"""Generate FastWAM RGB-only + SG-WAM cross-attention maps over MANY LIBERO samples, save to npz.
Selection/rendering is done separately (cheap) from the saved maps."""
import os, sys, functools, hashlib, torch, numpy as np
torch.load = functools.partial(torch.load, weights_only=False)
FW="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/FastWAM"
COSMOS="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/cosmos-predict2.5"
for p in (COSMOS, FW+"/src", FW+"/experiments/libero"):
    if p not in sys.path: sys.path.insert(0,p)
from fastwam.models.cosmos.runtime import create_fastwam_cosmos
W="/data/LFT-W02_data/junjie/weights/Cosmos-Predict2.5-2B"
CKPT="/data/LFT-W02_data/junjie/fastwam_sg_ckpt/checkpoints/weights/step_015000.pt"
QCACHE="/data/LFT-W02_data/junjie/fastwam_sg_ckpt/libero_qwen"
TPL="A video recorded from a robot's point of view executing the following instruction: {task}"
dev="cuda"; dt=torch.bfloat16; SEL_BLOCKS=[8,12,16,20]; TF=5
NSAMP=int(os.environ.get("NSAMP",240)); PER_TASK=int(os.environ.get("PER_TASK",6))
SAMPLES=os.environ.get("SAMPLES_NPZ","/data/LFT-W02_data/junjie/fastwam_sg_ckpt/libsamples_init.npz")
SUF=os.environ.get("OUT_SUFFIX","")   # e.g. "_main" -> wam_part_rgb_main.npz

def build(load_sg):
    m=create_fastwam_cosmos(video_dit_pretrained_path=W+"/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
        vae={"vae_pth":W+"/tokenizer.pth"},action_dim=7,proprio_dim=8,crossattn_dim=1024,coupling="mot",
        feature_layer=-1,action_hidden_dim=1024,action_ffn_dim=4096,action_attention_head_dim=128,model_dtype=dt,device=dev)
    if load_sg: m.load_checkpoint(CKPT)
    else:
        ck=torch.load(CKPT,map_location="cpu",weights_only=False); m.text_proj.load_state_dict(ck["text_proj"])
    return m.eval()

CAP={}
def install_capture(m, tmh):
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
                    mm=tmh["m"].float()[:,None,None,:]
                    CAP[idx]=((probs*mm).sum(-1)/(mm.sum(-1)+1e-6)).mean(1).float().cpu()
            return orig(x, context, rope_emb=rope_emb, **kw)
        cross.forward=fwd
    for i,blk in enumerate(net.blocks):
        if hasattr(blk,"cross_attn"): wrap(i, blk.cross_attn)

@torch.no_grad()
def vae_latents(m, comp):
    x=torch.from_numpy(np.ascontiguousarray(comp)).permute(2,0,1).float()/255.
    x=(x*2-1)[None,:,None].repeat(1,1,TF,1,1).to(dev,dt)
    return m._vae_encode(x).to(dt)
def load_ctx(prompt):
    h=hashlib.sha256(TPL.format(task=prompt).encode()).hexdigest()+".t5_len128.wan22ti2v5b.pt"
    d=torch.load(os.path.join(QCACHE,h),map_location="cpu",weights_only=False)
    return d["context"].to(dev,dt)[None], d["mask"].to(dev)[None]
@torch.no_grad()
def attn_map(m, comp, prompt, tmh):
    lat=vae_latents(m, comp); ctx,mask=load_ctx(prompt); tmh["m"]=mask
    crossattn=m.text_proj(ctx); Tl=lat.shape[2]
    t=torch.full((1,Tl),500.,device=dev,dtype=dt); CAP.clear()
    m.dit["video"].forward_standalone(lat, t, crossattn)
    maps=torch.stack([CAP[b] for b in SEL_BLOCKS]).mean(0)[0]
    Hl,Wl=lat.shape[-2]//2,lat.shape[-1]//2  # DiT patchifies latent 2x; grid dynamic (main-only -> 14x14)
    return maps.reshape(Tl,Hl,Wl)[0].numpy()

d=np.load(SAMPLES,allow_pickle=True)
CUR=d["cur"]; PROMPTS=[str(x) for x in d["prompts"]]
# pick NSAMP: distinct prompts first (up to 40), then extra frames for variety
import random; random.seed(5); order=list(range(len(CUR))); random.shuffle(order)
seen={}; picks=[]
for i in order:
    k=PROMPTS[i]
    if seen.get(k,0)>=PER_TASK: continue     # up to PER_TASK init-frame scenes per task
    seen[k]=seen.get(k,0)+1; picks.append(i)
    if len(picks)>=NSAMP: break
print("nsamples:",len(picks),flush=True)

which=sys.argv[1] if len(sys.argv)>1 else "rgb"   # 'rgb' or 'sg' — run ONE model per process (isolation)
load_sg = which=="sg"; tag="SG-WAM" if load_sg else "RGB-only"
print(f"building {tag}...",flush=True); m=build(load_sg); tmh={"m":None}; install_capture(m,tmh)
OUTP=f"/data/LFT-W02_data/junjie/fastwam_sg_ckpt/wam_part_{which}{SUF}.npz"
out=[]
if os.path.exists(OUTP):  # RESUME: reuse already-computed maps (same picks order), compute the rest
    prev=np.load(OUTP,allow_pickle=True)["maps"]
    out=[prev[j] for j in range(min(len(prev),len(picks)))]
    print(f"resume from {len(out)}/{len(picks)}",flush=True)
def save(k):
    np.savez(OUTP, maps=np.stack(out[:k]), frames=np.stack([CUR[i] for i in picks[:k]]),
        prompts=np.array([PROMPTS[i] for i in picks[:k]],dtype=object))
for n in range(len(out), len(picks)):
    out.append(attn_map(m, CUR[picks[n]], PROMPTS[picks[n]], tmh))
    if n%20==0: torch.cuda.empty_cache()
    if n%20==19: save(n+1); print(f"  {tag} {n+1}/{len(picks)} (checkpoint saved)",flush=True)
    elif n%10==0: print(f"  {tag} {n}/{len(picks)}",flush=True)
save(len(out))
print(f"SAVED wam_part_{which}.npz",np.stack(out).shape,flush=True); print("ALLDONE",flush=True)
