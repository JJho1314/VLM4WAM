import os, numpy as np
D=os.environ["DSDIR"]
rows=[]
with open(f"{D}/frame_ranges_audit.tsv") as f:
    next(f)
    for line in f:
        s,T,on,st,kept,status=line.rstrip("\n").split("\t")
        rows.append((s,int(T),int(on),int(st)))
T=np.array([r[1] for r in rows]); on=np.array([r[2] for r in rows])
active=T-on  # frames from onset to end
print(f"total {len(rows)}")
print("=== how many clips have very short active span (T-onset) ===")
for thr in [20,33,49,60,80]:
    m=active<thr
    print(f"  active < {thr:3d} frames: {int(m.sum()):4d}  ({m.mean()*100:.2f}%)")
print("=== also require the clip be 'mostly static': onset/T high ===")
for fr in [0.80,0.85,0.90,0.95]:
    m=(on/np.maximum(T,1))>=fr
    print(f"  onset/T >= {fr:.2f}: {int(m.sum()):4d} ({m.mean()*100:.2f}%)")
# the recommended rule: active < 49 (cannot fill a motion-only 49-frame clip)
m=active<49
print(f"\nRECOMMENDED exclude (active<49): {int(m.sum())} clips")
ex=sorted([r for r,a in zip(rows,active) if a<49], key=lambda r:(r[1]-r[2]))
for r in ex[:15]:
    print(f"  {r[0]:34s} T={r[1]:4d} onset={r[2]:4d} active={r[1]-r[2]:3d}")
