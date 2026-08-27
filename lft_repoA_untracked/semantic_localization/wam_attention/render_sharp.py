"""Re-render the FastWAM RGB-only vs SG-WAM poster from saved maps with contrast post-processing
(percentile clip + gamma) so the semantic-guided focus stands out. No model rerun."""
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
OUT="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/semantic_localization/figs"
d=np.load("/data/LFT-W02_data/junjie/fastwam_sg_ckpt/wam_maps.npz",allow_pickle=True)
RGB,SG,FR,PR=d["rgb"],d["sg"],d["frames"],[str(x) for x in d["prompts"]]
N=len(FR); TURBO=cm.get_cmap("turbo")
def sharp(g,lo=60,hi=99,gamma=1.6):
    g=(g-g.min())/(g.max()-g.min()+1e-6)
    plo,phi=np.percentile(g,[lo,hi]); x=np.clip((g-plo)/(phi-plo+1e-6),0,1); return x**gamma
def ov(comp,grid):
    up=F.interpolate(torch.from_numpy(sharp(grid))[None,None].float(),(224,448),mode="bilinear",align_corners=False)[0,0].numpy()
    return comp.astype(float)/255.*0.5+TURBO(up)[...,:3]*0.5
fig,ax=plt.subplots(2,N,figsize=(3.3*N,7))
for r,(tag,M) in enumerate((("RGB-only WAM",RGB),("SG-WAM (ours)",SG))):
    for c in range(N):
        ax[r,c].imshow(ov(FR[c],M[c])); ax[r,c].axis("off")
        if r==0: ax[r,c].set_title(PR[c][:32],fontsize=8)
    ax[r,0].axis("on"); ax[r,0].set_xticks([]); ax[r,0].set_yticks([])
    ax[r,0].set_ylabel(tag,fontsize=13,fontweight="bold")
fig.suptitle("FastWAM cross-attention on target instruction: RGB-only vs Semantic-Guided (LIBERO)",fontsize=14)
fig.tight_layout(); fig.savefig(f"{OUT}/wam_rgbonly_vs_sg.png",dpi=110,bbox_inches="tight"); plt.close(fig)
print("SAVED",flush=True)
