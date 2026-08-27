import json, numpy as np, os
D=os.environ["DSDIR"]
J=json.load(open(f"{D}/frame_ranges.candidate.json"))
print(f"json entries (trimmed episodes): {len(J)}")
# validate format
bad=0
for k,v in J.items():
    if not (isinstance(v,list) and len(v)==1 and len(v[0])==2 and 0<=v[0][0]<v[0][1]):
        bad+=1
print(f"malformed entries: {bad}")
ex=next(iter(J.items())); print("sample entry:", ex)

# audit buckets
rows=[]
with open(f"{D}/frame_ranges_audit.tsv") as f:
    next(f)
    for line in f:
        s,T,on,st,kept,status=line.rstrip("\n").split("\t")
        rows.append((s,int(T),int(on),int(st),int(kept),status))
trim=np.array([on-0 for _,T,on,st,_,_ in rows])  # onset
starts=np.array([st for *_,st,_,_ in [ (r[0],r[1],r[2],r[3],r[4],r[5]) for r in rows]])
st_arr=np.array([r[3] for r in rows]); T_arr=np.array([r[1] for r in rows]); kept_arr=np.array([r[4] for r in rows])

print("\n=== trim(start) size buckets ===")
for lo,hi in [(0,0),(1,5),(6,15),(16,30),(31,60),(61,120),(121,10**9)]:
    if lo==hi==0: m=(st_arr==0)
    else: m=(st_arr>=lo)&(st_arr<=hi)
    label=f"{lo}" if lo==hi else f"{lo}-{(hi if hi<10**9 else '+')}"
    print(f"  start={label:8s}: {int(m.sum()):6d}  ({m.mean()*100:4.1f}%)")

print("\n=== kept-frames buckets (after trim) ===")
for lo,hi in [(49,49),(50,96),(97,10**9)]:
    m=(kept_arr>=lo)&(kept_arr<=hi)
    print(f"  kept {lo}-{hi if hi<10**9 else '+':>3}: {int(m.sum()):6d} ({m.mean()*100:4.1f}%)")

print("\n=== examples: no-trim (start==0) ===")
import itertools
for r in [r for r in rows if r[3]==0][:5]:
    print(f"  {r[0]:34s} T={r[1]:4d} onset={r[2]:3d} start={r[3]:3d} kept={r[4]}")
print("=== examples: small trim (5..15) ===")
for r in [r for r in rows if 5<=r[3]<=15][:5]:
    print(f"  {r[0]:34s} T={r[1]:4d} onset={r[2]:3d} start={r[3]:3d} kept={r[4]}")
print("=== examples: large trim (>=80) ===")
big=sorted([r for r in rows if r[3]>=80], key=lambda r:-r[3])[:8]
for r in big:
    print(f"  {r[0]:34s} T={r[1]:4d} onset={r[2]:3d} start={r[3]:3d} kept={r[4]}")
