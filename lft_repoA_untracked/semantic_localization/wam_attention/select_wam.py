"""Score each sample by how much MORE focused SG-WAM is than RGB-only (concentration gain),
rank, render the top-K clearest cases as the 2-row poster. Cheap: works from saved maps."""
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
import os
B="/data/LFT-W02_data/junjie/fastwam_sg_ckpt"
OUT="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/figs"
SUF=os.environ.get("OUT_SUFFIX","")        # "_main" -> reads wam_part_*_main.npz, writes wam_sg_best_main.png
FIGNAME=os.environ.get("FIGNAME",f"wam_sg_best{SUF or '_init'}")
r=np.load(f"{B}/wam_part_rgb{SUF}.npz",allow_pickle=True); s=np.load(f"{B}/wam_part_sg{SUF}.npz",allow_pickle=True)
N=min(len(r["maps"]),len(s["maps"]))  # rgb/sg share the same sample order; align to the shorter
RGB,SG,FR,PR=r["maps"][:N],s["maps"][:N],r["frames"][:N],[str(x) for x in r["prompts"]][:N]
TOPK=int(os.environ.get("TOPK",10))
print(f"paired samples: {N}",flush=True)

def conc(g):  # top-10% mass fraction of a normalized map (higher = more focused)
    v=g.flatten().astype(np.float64); v=v-v.min(); p=v/(v.sum()+1e-9)
    k=max(1,int(0.10*len(p))); return float(np.sort(p)[::-1][:k].sum())
def edge_frac(g):  # mass on the 1-patch border (penalize edge-artifact peaks)
    gg=(g-g.min()); gg=gg/(gg.sum()+1e-9); b=gg.copy(); b[1:-1,1:-1]=0; return float(b.sum())

scores=[]
for i in range(N):
    cs,cr=conc(SG[i]),conc(RGB[i])
    gain=cs-cr
    pen=0.5*edge_frac(SG[i])                 # discourage SG peaks stuck on the frame border
    scores.append((gain-pen, cs, cr, i))
scores.sort(reverse=True)
# diverse top: at most 1 per distinct task so the poster spans many scenes
seenp=set(); top=[]
for sc,cs,cr,i in scores:
    if PR[i] in seenp: continue
    seenp.add(PR[i]); top.append(i)
    print(f"  {sc:+.3f}  SG={cs:.3f} RGB={cr:.3f}  {PR[i][:44]}",flush=True)
    if len(top)>=TOPK: break

TURBO=cm.get_cmap("turbo")
def sharp(g,lo=60,hi=99,gamma=1.6):
    g=(g-g.min())/(g.max()-g.min()+1e-6); plo,phi=np.percentile(g,[lo,hi])
    return (np.clip((g-plo)/(phi-plo+1e-6),0,1))**gamma
def ov(comp,g):
    H,W=comp.shape[:2]  # upsample the attn map to the frame's own size (main-only 224x224 or composite 224x448)
    up=F.interpolate(torch.from_numpy(sharp(g))[None,None].float(),(H,W),mode="bilinear",align_corners=False)[0,0].numpy()
    return comp.astype(float)/255.*0.5+TURBO(up)[...,:3]*0.5

fig,ax=plt.subplots(2,TOPK,figsize=(3.3*TOPK,7))
for r_,(tag,M) in enumerate((("RGB-only WAM",RGB),("SG-WAM (ours)",SG))):
    for c,i in enumerate(top):
        ax[r_,c].imshow(ov(FR[i],M[i])); ax[r_,c].axis("off")
        if r_==0: ax[r_,c].set_title(PR[i][:30],fontsize=8)
    ax[r_,0].axis("on"); ax[r_,0].set_xticks([]); ax[r_,0].set_yticks([])
    ax[r_,0].set_ylabel(tag,fontsize=13,fontweight="bold")
fig.suptitle("FastWAM cross-attention — clearest RGB-only(diffuse) vs SG-WAM(focused) cases (LIBERO, main view)",fontsize=14)
fig.tight_layout(); fig.savefig(f"{OUT}/{FIGNAME}.png",dpi=110,bbox_inches="tight"); plt.close(fig)
print(f"SAVED {FIGNAME}.png  top prompts:",[PR[i][:24] for i in top],flush=True)
