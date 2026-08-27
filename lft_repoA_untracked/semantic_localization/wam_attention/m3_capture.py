"""Milestone 3: verify cross-attention capture works. Monkeypatch each video block's
cross_attn to record attention from video tokens to valid text tokens, run forward_standalone
on dummy latents, print captured map shapes."""
import os, sys, functools, torch
torch.load = functools.partial(torch.load, weights_only=False)
FW="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/FastWAM"
COSMOS="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/cosmos-predict2.5"
for p in (COSMOS, FW+"/src", FW+"/experiments/libero"):
    if p not in sys.path: sys.path.insert(0,p)
from fastwam.models.cosmos.runtime import create_fastwam_cosmos
W="/data/LFT-W02_data/junjie/weights/Cosmos-Predict2.5-2B"
dev="cuda"; dt=torch.bfloat16
model=create_fastwam_cosmos(video_dit_pretrained_path=W+"/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
    vae={"vae_pth":W+"/tokenizer.pth"},action_dim=7,proprio_dim=8,crossattn_dim=1024,coupling="mot",
    feature_layer=-1,action_hidden_dim=1024,action_ffn_dim=4096,action_attention_head_dim=128,model_dtype=dt,device=dev)
model.load_checkpoint("/data/LFT-W02_data/junjie/fastwam_sg_ckpt/checkpoints/weights/step_015000.pt")
model=model.eval()
ve=model.dit["video"]; net=ve.net
print("nblocks:",len(net.blocks),flush=True)

CAP={}; TEXTMASK={"m":None}
def wrap(blk_idx, cross):
    orig=cross.forward
    def fwd(x, context=None, rope_emb=None, **kw):
        if context is not None:
            with torch.no_grad():
                q,k,v=cross.compute_qkv(x, context, rope_emb=rope_emb)  # [B,S,H,D],[B,L,H,D]
                scale=1.0/(q.shape[-1]**0.5)
                # attention video->text, mean over heads
                scores=torch.einsum("bshd,blhd->bhsl", q.float(), k.float())*scale
                probs=scores.softmax(dim=-1)  # [B,H,S,L]
                m=TEXTMASK["m"]
                if m is not None:
                    w=m.float()[:,None,None,:]  # [B,1,1,L]
                    amap=(probs*w).sum(-1)/(w.sum(-1)+1e-6)  # [B,H,S]
                else:
                    amap=probs.mean(-1)
                CAP[blk_idx]=amap.mean(1).float().cpu()  # [B,S] mean over heads
        return orig(x, context, rope_emb=rope_emb, **kw)
    cross.forward=fwd
for i,blk in enumerate(net.blocks):
    if hasattr(blk,"cross_attn"): wrap(i, blk.cross_attn)

# dummy inputs: latents [B,16,T,Hl,Wl]; forward_standalone adds conditioning channel
B,T,Hl,Wl=1,6,28,28
lat=torch.randn(B,16,T,Hl,Wl,device=dev,dtype=dt)
ctx=torch.randn(B,128,3584,device=dev,dtype=dt)
crossattn=model.text_proj(ctx)
TEXTMASK["m"]=torch.ones(B,128,dtype=torch.bool,device=dev)
t=torch.full((B,T),500.0,device=dev,dtype=dt)
with torch.no_grad():
    ve.forward_standalone(lat, t, crossattn)
print("captured blocks:",sorted(CAP.keys())[:6],"...",len(CAP),flush=True)
for i in (8,12,16,20):
    if i in CAP: print(f"  block {i}: attn map shape {tuple(CAP[i].shape)}  S={CAP[i].shape[-1]} (=T*Hl*Wl={T*Hl*Wl})",flush=True)
print("M3-CAPTURE-OK",flush=True)
