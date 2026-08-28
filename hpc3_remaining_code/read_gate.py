import torch, math, sys
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
for k, v in sorted(sd.items()):
    if "dense_spatial" in k:
        if v.numel() == 1:
            g = float(v.float().item())
            print(f"GATE {k} = {g:.6f}  tanh={math.tanh(g):.6f}  (init 0.01)")
        else:
            print(f"PARAM {k}: shape={tuple(v.shape)} norm={float(v.float().norm()):.3f}")
