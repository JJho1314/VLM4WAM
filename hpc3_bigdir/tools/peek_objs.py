import os,sys,numpy as np,torch
REPO="/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5"; SRC="/data/user/jhe724/workspace/InstructSAM"
for p in (f"{REPO}/scripts/_env_stubs",REPO,SRC):
    sys.path.insert(0,p)
# just reload saved maps to inspect
OUT="/data/user/jhe724/workspace/VLM4WAM/feature_guidance_analysis/instructsam_query_match_74616"
for n in ["carrot","banana"]:
    m=np.load(f"{OUT}/{n}_softmap_full.npy")
    print(n,"full softmap: shape",m.shape,"min",round(float(m.min()),3),"max",round(float(m.max()),3),
          "frac>0.5",round(float((m>0.5).mean()),5),"frac>0.3",round(float((m>0.3).mean()),5))
