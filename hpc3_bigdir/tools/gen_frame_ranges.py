#!/usr/bin/env python
"""Generate frame_ranges.json marking each episode's motion onset so the training
loader (frame_start_policy='range_start') skips the static lead-in.

Decoding: bundled ffmpeg -> downscaled grayscale rawvideo (fast, C-side scaling).
Resumable: each result is appended to frame_ranges_progress.ndjson; a restart skips
stems already recorded. When the full set is done it writes frame_ranges.candidate.json,
frame_ranges_audit.tsv and a SUMMARY.
"""
import os, sys, glob, json, subprocess
from multiprocessing import Pool
import numpy as np
import imageio_ffmpeg

D = os.environ["DSDIR"]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
DW, DH = 432, 240        # decode resolution (matches earlier [::2,::2] of 864x480)
PIX_THR   = 12
SMOOTH_W  = 5
ABS_FLOOR = 1.0
REL       = 0.20
MARGIN    = 4
MIN_KEEP  = 49
WORKERS   = int(os.environ.get("WORKERS", "12"))
NDJSON    = f"{D}/frame_ranges_progress.ndjson"

def iter_gray(path):
    cmd = [FFMPEG, "-v", "error", "-i", path, "-vf", f"scale={DW}:{DH}",
           "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    fsz = DW * DH
    try:
        while True:
            buf = p.stdout.read(fsz)
            if len(buf) < fsz:
                break
            yield np.frombuffer(buf, np.uint8).reshape(DH, DW).astype(np.float32)
    finally:
        p.stdout.close(); p.wait()

def compute(stem):
    path = f"{D}/videos/{stem}.mp4"
    try:
        prev = None; sc = []
        for g in iter_gray(path):
            if prev is not None:
                sc.append(float((np.abs(g - prev) > PIX_THR).mean()) * 100.0)
            prev = g
        if not sc:
            return {"stem": stem, "T": (1 if prev is not None else 0), "onset": 0, "start": 0, "status": "short"}
        frac = np.asarray(sc, dtype=np.float32)
        T = len(frac) + 1
        sm = np.convolve(frac, np.ones(SMOOTH_W)/SMOOTH_W, mode="same")
        thr = max(ABS_FLOOR, REL*float(np.percentile(sm, 95)))
        act = np.where(sm >= thr)[0]
        onset = (int(act[0]) + 1) if len(act) else 0
        start = min(max(0, onset - MARGIN), max(0, T - MIN_KEEP))
        return {"stem": stem, "T": int(T), "onset": int(onset), "start": int(start), "status": "ok"}
    except Exception as e:
        return {"stem": stem, "T": -1, "onset": -1, "start": 0, "status": f"ERR:{type(e).__name__}"}

def finalize():
    rows = {}
    with open(NDJSON) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line); rows[r["stem"]] = r
    ranges = {}; audit = ["stem\tT\tonset\tstart\tkept\tstatus"]
    n_trim = n_zero = n_err = 0; trims = []
    for stem, r in sorted(rows.items()):
        T, onset, start, status = r["T"], r["onset"], r["start"], r["status"]
        audit.append(f"{stem}\t{T}\t{onset}\t{start}\t{T-start if T>0 else 0}\t{status}")
        if status.startswith("ERR") or T <= 0: n_err += 1; continue
        if start > 0:
            ranges[stem] = [[start, T-1]]; n_trim += 1; trims.append(start)
        else: n_zero += 1
    with open(f"{D}/frame_ranges.candidate.json", "w") as f: json.dump(ranges, f)
    with open(f"{D}/frame_ranges_audit.tsv", "w") as f: f.write("\n".join(audit))
    t = np.asarray(trims) if trims else np.zeros(1)
    print("================ SUMMARY ================", flush=True)
    print(f"episodes recorded   : {len(rows)}")
    print(f"trimmed (start>0)   : {n_trim}  ({n_trim/max(len(rows),1)*100:.1f}%)")
    print(f"no trim (start==0)  : {n_zero}")
    print(f"errors/too-short    : {n_err}")
    print(f"trim frames: med={np.median(t):.0f} mean={t.mean():.1f} p90={np.percentile(t,90):.0f} p99={np.percentile(t,99):.0f} max={t.max():.0f}")
    print(f"candidate json -> {D}/frame_ranges.candidate.json")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        stems = [os.path.basename(p)[:-4] for p in sorted(glob.glob(f"{D}/videos/*.mp4"))]
        idx = np.linspace(0, len(stems)-1, 12).astype(int)
        for i in idx:
            r = compute(stems[i])
            print(f"{r['stem']:34s} T={r['T']:4d} onset={r['onset']:3d} start={r['start']:3d} kept={r['T']-r['start']}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        finalize(); return

    stems = [os.path.basename(p)[:-4] for p in sorted(glob.glob(f"{D}/videos/*.mp4"))]
    done = set()
    if os.path.exists(NDJSON):
        with open(NDJSON) as f:
            for line in f:
                line = line.strip()
                if line:
                    try: done.add(json.loads(line)["stem"])
                    except Exception: pass
    todo = [s for s in stems if s not in done]
    print(f"total={len(stems)} done={len(done)} todo={len(todo)} workers={WORKERS}", flush=True)
    out = open(NDJSON, "a", buffering=1)
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(compute, todo, chunksize=8)):
            out.write(json.dumps(r) + "\n")
            if (i+1) % 2000 == 0:
                print(f"  processed {i+1}/{len(todo)}", flush=True)
    out.close()
    if len([s for s in stems if s not in done]) and len(done)+len(todo) >= len(stems):
        finalize()

if __name__ == "__main__":
    main()
