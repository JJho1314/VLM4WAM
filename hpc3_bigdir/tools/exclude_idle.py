import os, json
D=os.environ["DSDIR"]
ACTIVE_THR=int(os.environ.get("ACTIVE_THR","20"))
rows=[]
with open(f"{D}/frame_ranges_audit.tsv") as f:
    next(f)
    for line in f:
        s,T,on,st,kept,status=line.rstrip("\n").split("\t")
        rows.append((s,int(T),int(on)))
idle=sorted([r for r in rows if (r[1]-r[2])<ACTIVE_THR], key=lambda r:(r[1]-r[2], r[0]))
stems=[r[0] for r in idle]
print(f"ACTIVE_THR={ACTIVE_THR}  near-idle clips to exclude: {len(stems)}")
for r in idle:
    print(f"  {r[0]:34s} T={r[1]:4d} onset={r[2]:4d} active={r[1]-r[2]}")

# 1) provenance file
with open(f"{D}/exclude_near_idle_stems.txt","w") as f:
    f.write("\n".join(stems)+"\n")

# 2) merge into the operative exclude file (preserve existing, de-dup)
op=f"{D}/exclude_no_tgt_stems.txt"
existing=[]
if os.path.exists(op):
    with open(op) as f:
        existing=[l.strip() for l in f if l.strip()]
merged=sorted(set(existing)|set(stems))
with open(op,"w") as f:
    f.write("\n".join(merged)+"\n")
print(f"\nexclude_no_tgt_stems.txt: existing={len(existing)} -> merged={len(merged)}")

# 3) drop excluded entries from the live frame_ranges.json (candidate keeps full backup)
fr=json.load(open(f"{D}/frame_ranges.json"))
before=len(fr)
for s in stems: fr.pop(s, None)
json.dump(fr, open(f"{D}/frame_ranges.json","w"))
print(f"frame_ranges.json: {before} -> {len(fr)} entries (removed excluded)")
print(f"\nwrote: exclude_near_idle_stems.txt, updated exclude_no_tgt_stems.txt, frame_ranges.json")
