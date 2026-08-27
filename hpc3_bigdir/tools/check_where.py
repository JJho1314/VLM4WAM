import torch, numpy as np, glob, os
D="/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half/target_features_where_softlogit_stage2_lora_TEST"
for p in sorted(glob.glob(f"{D}/*.pt"))[:4]:
    d=torch.load(p,map_location="cpu",weights_only=False)
    wl=d["where_logit"]; wp=d["where_prob"]
    # concentration: top-k fraction of prob mass
    flat=wp.flatten(); k=int(0.05*flat.numel())
    topfrac=float(flat.topk(k).values.sum()/flat.sum().clamp_min(1e-6))
    print(f"{os.path.basename(p):34s} grid={d['grid']} peak_logit={d['peak_logit']:+.2f} slot_cls={d['slot_cls']:+.3f} "
          f"prob[min={float(wp.min()):.3f} max={float(wp.max()):.3f}] top5%massfrac={topfrac:.2f}")
    print(f"     query={d['query']!r}")
