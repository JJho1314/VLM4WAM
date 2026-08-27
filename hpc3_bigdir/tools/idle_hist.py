import os, numpy as np
D=os.environ["DSDIR"]
rows=[]
with open(f"{D}/frame_ranges_audit.tsv") as f:
    next(f)
    for line in f:
        s,T,on,st,kept,status=line.rstrip("\n").split("\t")
        rows.append((s,int(T),int(on)))
active=np.array([r[1]-r[2] for r in rows])
print("histogram of active span (T-onset) for active<70:")
for lo in range(0,70,5):
    m=(active>=lo)&(active<lo+5)
    bar="#"*int(m.sum())
    print(f"  [{lo:2d},{lo+5:2d}): {int(m.sum()):4d} {bar[:60]}")
# precise small end
print("\nexact small-active counts:")
for a in range(0,15):
    c=int((active==a).sum())
    if c: print(f"  active={a:2d}: {c}")
# onset/T view among active<49
T=np.array([r[1] for r in rows]); on=np.array([r[2] for r in rows])
print("\namong active<49, distribution of onset/T:")
sub=(active<49)
fr=on[sub]/np.maximum(T[sub],1)
for lo in [0,0.3,0.5,0.7,0.85,0.95]:
    print(f"  onset/T>={lo:.2f}: {int((fr>=lo).sum())}")
