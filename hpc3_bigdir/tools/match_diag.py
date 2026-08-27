import torch, numpy as np
BASE="/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets"
CD=f"{BASE}/robointer_74616_yellow_carrot_prompt_targetaware_dataset"
BD=f"{BASE}/robointer_74616_banana_prompt_targetaware_dataset"
def load_dense(p):
    return torch.load(p, map_location="cpu", weights_only=False)["target_feature"].float()
carrot=load_dense(f"{CD}/target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt_s20260613/74616_exterior_image_1_left.pt")
banana=load_dense(f"{BD}/target_features_instructsam_decoder_dense_stage2_lora_banana_prompt_s20260613/74616_exterior_image_1_left.pt")
g=int(round(carrot.shape[0]**0.5))
def gcos(a,b):
    a=a.flatten(); b=b.flatten(); return float(torch.dot(a,b)/(a.norm()*b.norm()))
print("grid",g,"x",g,"| global cosine carrot vs banana dense:", round(gcos(carrot,banana),4))

def load_mask(p,g):
    z=np.load(p, allow_pickle=True)
    arr=z[z.files[0]]
    s0=arr.shape
    arr=np.asarray(arr)
    while arr.ndim>2:            # collapse leading dims -> union over frames/instances
        arr=arr.max(axis=0)
    m=(arr>0).astype(np.float32)
    H,W=m.shape
    md=torch.nn.functional.adaptive_avg_pool2d(torch.tensor(m).view(1,1,H,W),(g,g)).view(g,g).numpy()
    return s0,(H,W),(md>0.25).astype(np.float32)
cs,cHW,cm=load_mask(f"{CD}/target_masks/74616_exterior_image_1_left.npz",g)
bs,bHW,bm=load_mask(f"{BD}/target_masks/74616_exterior_image_1_left.npz",g)
print(f"carrot mask npz shape={cs} -> HxW={cHW} area32={cm.mean():.4f}")
print(f"banana mask npz shape={bs} -> HxW={bHW} area32={bm.mean():.4f}")
print("mask IoU carrot vs banana (32x32):", round(float((cm*bm).sum()/np.clip((cm+bm-cm*bm).sum(),1,None)),4))

def heat(dense,mask2d):
    D=dense.view(g,g,-1); Dn=torch.nn.functional.normalize(D,dim=-1)
    m=torch.tensor(mask2d)
    proto=torch.nn.functional.normalize((D*m.unsqueeze(-1)).sum((0,1))/max(float(m.sum()),1.0),dim=0)
    return (Dn*proto).sum(-1)
hc=heat(carrot,cm); hb=heat(banana,bm)
def rm(h,mask): mm=torch.tensor(mask).bool(); return float(h[mm].mean()), float(h[~mm].mean())
ic,oc=rm(hc,cm); ib,ob=rm(hb,bm)
print(f"\n[carrot query · carrot dense] inside carrot mask={ic:.3f} outside={oc:.3f} lift={ic-oc:+.3f}")
print(f"[banana query · banana dense] inside banana mask={ib:.3f} outside={ob:.3f} lift={ib-ob:+.3f}")
print(f"\ncarrot heatmap: on carrot region={float(hc[torch.tensor(cm).bool()].mean()):.3f}  on banana region={float(hc[torch.tensor(bm).bool()].mean()):.3f}")
print(f"banana heatmap: on banana region={float(hb[torch.tensor(bm).bool()].mean()):.3f}  on carrot region={float(hb[torch.tensor(cm).bool()].mean()):.3f}")
a=hc.flatten()-hc.mean(); b=hb.flatten()-hb.mean()
print("\nspatial corr(carrot_heatmap, banana_heatmap):", round(float((a*b).sum()/(a.norm()*b.norm())),4))
np.savez("/data/user/jhe724/workspace/VLM4WAM/tools/heatmaps.npz", carrot=hc.numpy(), banana=hb.numpy(), cm=cm, bm=bm, g=g)
print("saved -> tools/heatmaps.npz")
