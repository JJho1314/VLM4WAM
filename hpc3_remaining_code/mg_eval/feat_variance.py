import torch, glob, os
import torch.nn.functional as F

DS = "/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half"

def pooled(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    feat = p["target_feature"].float()
    valid = feat.abs().sum(-1, keepdim=True) > 0
    return (feat * valid).sum(0) / valid.sum(0).clamp_min(1).float()

def interstats(X, name):
    Xn = F.normalize(X, dim=-1)
    S = Xn @ Xn.t(); n = S.shape[0]
    off = (S.sum() - S.diag().sum()) / (n * n - n)
    c = F.normalize(X.mean(0, keepdim=True), dim=-1)
    cos_c = (Xn @ c.t()).mean()
    print(f"{name}: N={n} dim={X.shape[1]} | mean_pairwise_cos={off:.4f} | mean_cos_to_centroid={cos_c:.4f} | perdim_std_mean={X.std(0).mean():.4f}", flush=True)

stems = sorted(os.path.basename(p)[:-3] for p in glob.glob(f"{DS}/target_features_ft/*.pt") if not p.endswith("_mask.png"))
stems = [s for s in stems if not s.endswith("_mask")][:400]
Q3 = torch.stack([pooled(f"{DS}/target_features_ft/{s}.pt") for s in stems])
Q2 = torch.stack([pooled(f"{DS}/target_features_rawseg_ft/{s}.pt") for s in stems])
interstats(Q3, "feat3_proj256")
interstats(Q2, "feat2_raw2048")
print("DONE", flush=True)
