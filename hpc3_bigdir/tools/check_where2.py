import torch, glob, os
D="/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half/target_features_where_softlogit_stage2_lora_TEST2"
fs=sorted(glob.glob(f"{D}/*.pt"))
print("files:",len(fs))
for p in fs[:3]:
    d=torch.load(p,map_location="cpu",weights_only=False)
    def shp(k):
        v=d.get(k); return None if v is None else (tuple(v.shape),str(v.dtype))
    print(f"\n{os.path.basename(p)}")
    print("  where_logit",shp("where_logit"),"where_prob",shp("where_prob"))
    print("  target_dense_weighted",shp("target_dense_weighted"),"target_proto",shp("target_proto"))
    wp=d["where_prob"]; print(f"  prob max={float(wp.max()):.3f} peak_logit={d['peak_logit']:+.2f} slot_cls={d['slot_cls']:+.3f}")
    tw=d.get("target_dense_weighted")
    if tw is not None:
        tw=tw.float(); print(f"  weighted: nonzero_frac={float((tw.abs().sum(-1)>1e-4).float().mean()):.3f} norm={float(tw.norm()):.2f}")
    print(f"  query={d['query']!r}")
