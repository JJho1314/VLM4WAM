import os,sys,json,argparse,random
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
for p in (HERE,os.path.join(HERE,"dinov3_da3_2b")): sys.path.insert(0,p)
from train_feature_probes import ProbeDecoder,MiniDPTProbe,depth_to_disp
from depth_anything3_target import _import_da3,DepthAnything3TargetEncoder
def nz(a): return (a-a.min())/(a.max()-a.min()+1e-6)
def met(pred,gt,eps=1e-3):
    pred,gt=pred.flatten().float(),gt.flatten().float()
    return ((pred-gt).abs()/(gt+eps)).mean().item(), torch.sqrt(((pred-gt)**2).mean()).item(), torch.corrcoef(torch.stack([pred,gt]))[0,1].item()
device="cuda"
da3=DepthAnything3TargetEncoder(process_res=224,device=device)
DA3=_import_da3(os.environ["DA3_CODE_ROOT"]); full=DA3.from_pretrained(os.environ["DA3_CKPT_DIR"]).to(device).eval()
for p in full.parameters(): p.requires_grad_(False)
m=full.model
v1=torch.load("/data/users/junjie/probes_2b/da3_depth_probe.pt",map_location="cpu",weights_only=False)
p1=ProbeDecoder(**v1["config"]).to(device).eval(); p1.load_state_dict(v1["state_dict"])
v2=torch.load("/data/users/junjie/probes_2b/da3_depth_v2_probe.pt",map_location="cpu",weights_only=False)
p2=MiniDPTProbe(**v2["config"]).to(device).eval(); p2.load_state_dict(v2["state_dict"])
import glob
files=sorted(glob.glob("/data/users/junjie/data/frame_cache/libero/**/*.npy",recursive=True))
rng=random.Random(7); r1=[];r2=[]
for _ in range(8):
    f=files[rng.randrange(len(files))]; mm=np.load(f,mmap_mode="r"); fi=rng.randrange(mm.shape[0])
    fr=torch.from_numpy(np.ascontiguousarray(mm[fi])).float().unsqueeze(0)/255.0  # [1,3,224,224]
    fr=fr.to(device)
    with torch.no_grad():
        feats=da3._patch_tokens(da3._prep(fr)).float()
        x=da3._prep(fr).to(next(m.parameters()).dtype).unsqueeze(1)
        bf,_=m.backbone(x,cam_token=None,export_feat_layers=[],ref_view_strategy="saddle_balanced")
        with torch.autocast(device_type="cuda",enabled=False): out=m._process_depth_head(bf,224,224)
        gt=nz(depth_to_disp((out["depth"] if hasattr(out,"keys") else out.depth).float())[0,0])
        d1=nz(p1(feats)[0,0])
        d2=nz(1.0/torch.exp(p2(feats)[0,0]).clamp_min(1e-3))
    r1.append(met(d1,gt)); r2.append(met(d2,gt))
import statistics as st
def avg(rs,j): return st.mean(r[j] for r in rs)
print("\n==== probe-from-REAL vs DA3-full GT (normalized disparity, 8 frames) ====")
print(f"{'probe':<10}{'AbsRel':>8}{'RMSE':>8}{'corr':>8}")
print(f"{'v1(plain)':<10}{avg(r1,0):>8.3f}{avg(r1,1):>8.3f}{avg(r1,2):>8.3f}")
print(f"{'v2(miniDPT)':<10}{avg(r2,0):>8.3f}{avg(r2,1):>8.3f}{avg(r2,2):>8.3f}")
