import torch, glob, os, re
import torch.nn.functional as F
from collections import defaultdict

DS = "/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half"

def load(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    feat = p["target_feature"].float()
    valid = feat.abs().sum(-1, keepdim=True) > 0
    pooled = (feat * valid).sum(0) / valid.sum(0).clamp_min(1).float()
    obj = (p.get("target_phrase") or "").strip().lower()
    obj = re.sub(r"\b(the|a|an|one|two|some|left|right)\b", "", obj)
    obj = re.sub(r"[^a-z ]", " ", obj); obj = re.sub(r"\s+", " ", obj).strip()
    return pooled, obj

stems = sorted({os.path.basename(p)[:-3] for p in glob.glob(f"{DS}/target_features_ft/*.pt") if p.endswith('.pt')})
stems = [s for s in stems if os.path.exists(f"{DS}/target_features_rawseg_ft/{s}.pt")][:800]

for name, sub in [("feat3_proj256","target_features_ft"), ("feat2_raw2048","target_features_rawseg_ft")]:
    X, objs = [], []
    for s in stems:
        v, o = load(f"{DS}/{sub}/{s}.pt")
        if o: X.append(v); objs.append(o)
    X = torch.stack(X)
    def within_cross(M, tag):
        Mn = F.normalize(M, dim=-1)
        by = defaultdict(list)
        for i,o in enumerate(objs): by[o].append(i)
        wi, cr = [], []
        S = Mn @ Mn.t()
        for o, idx in by.items():
            if len(idx) < 2: continue
            for a in range(len(idx)):
                for b in range(a+1, len(idx)):
                    wi.append(S[idx[a], idx[b]].item())
        import random; random.seed(0)
        keys = list(by.keys())
        for _ in range(3000):
            o1, o2 = random.sample(keys, 2)
            cr.append(S[random.choice(by[o1]), random.choice(by[o2])].item())
        import statistics as st
        print(f"  {name} {tag}: within_obj_cos={st.mean(wi):.4f} cross_obj_cos={st.mean(cr):.4f} sep={st.mean(wi)-st.mean(cr):+.4f} (n_obj_groups={sum(1 for o in by.values() if len(o)>=2)})", flush=True)
    within_cross(X, "raw")
    within_cross(X - X.mean(0, keepdim=True), "centered")
print("DONE", flush=True)
