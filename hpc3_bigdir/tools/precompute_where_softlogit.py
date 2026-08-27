#!/usr/bin/env python
"""Precompute InstructSAM [SEG]·dense localization + target-aware dense features.

Per episode, saves:
  where_logit          [G,G]      raw soft mask-logit (discriminative localization)
  where_prob           [G,G]      sigmoid(where_logit) in 0..1
  target_dense_weighted[G*G,256]  decoder_dense grid * where_prob  (target-localized 256-d grid)
  target_proto         [256]      where-pooled target semantic vector
Query is taken from the existing decoder_dense .pt (alignment with the 'what' features).
"""
import os, sys, time, json, tempfile, importlib.util
from pathlib import Path
import torch, torch.nn.functional as F

REPO=os.environ["REPO_ROOT"]; SRC=os.environ["INSTRUCTSAM_SOURCE_ROOT"]; MODEL=os.environ["INSTRUCTSAM_MODEL_PATH"]
for p in (f"{REPO}/scripts/_env_stubs", REPO, SRC):
    if p not in sys.path: sys.path.insert(0,p)
spec=importlib.util.spec_from_file_location("isam_pc", f"{REPO}/scripts/precompute_instructsam_target_features.py")
P=importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
from cosmos_predict2._src.predict2.target_aware.instructsam_mask import (
    InstructSAMTargetMaskGenerator, read_first_frame_image)
from instructsam import mm_infer_segmentation

DATASET=Path(os.environ["DSDIR"])
DENSE_DIR=os.environ.get("DENSE_DIR_NAME","target_features_instructsam_decoder_dense_stage2_lora")
OUT_NAME=os.environ.get("OUT_DIR_NAME","target_features_where_softlogit_stage2_lora")
GRID=int(os.environ.get("WHERE_GRID","32")); LIMIT=int(os.environ.get("LIMIT","0"))
TEMPLATE=os.environ.get("QUERY_TEMPLATE","Please segment '{target}' in the image.")
FALLBACK=os.environ.get("FALLBACK_QUERY","Please segment the target object in the image.")

def query_for(stem, caption):
    dp=DATASET/DENSE_DIR/f"{stem}.pt"
    if dp.exists():
        try:
            d=torch.load(dp, map_location="cpu", weights_only=False)
            if d.get("query"): return d["query"], d.get("target_phrase")
        except Exception: pass
    return P.build_query(caption, TEMPLATE, FALLBACK)

def run(image_path, query):
    gen._decoder_dense_capture=[]                         # reset hook buffer
    output,pred_masks,cls_score=mm_infer_segmentation(image_path, gen.processor, [
        {"role":"user","content":[{"type":"image","image":image_path},{"type":"text","text":query}]}],
        gen.model, gen.tokenizer)
    if pred_masks is None: return None
    pm=pred_masks.detach().float().cpu()
    if pm.ndim==4: pm=pm[0]
    if pm.ndim==2: pm=pm[None]
    N=pm.shape[0]
    cs=(cls_score.detach().float().cpu().reshape(-1)[:N] if cls_score is not None else torch.zeros(N))
    best=int(cs.argmax()); raw=pm[best]
    logit=F.interpolate(raw[None,None], size=(GRID,GRID), mode="bilinear", align_corners=False)[0,0]   # G,G
    prob=torch.sigmoid(logit)
    # query-conditioned dense grid (the generator averages object queries -> [1,G*G,256])
    dense=gen._extract_target_feature(feature_mode="decoder_dense")
    weighted=proto=None
    if dense is not None:
        dg=dense.squeeze(0).float()                       # [G*G,256]
        side=int(round(dg.shape[0]**0.5))
        grid=dg.view(side,side,-1)
        if (side,side)!=(GRID,GRID):
            grid=F.interpolate(grid.permute(2,0,1)[None],size=(GRID,GRID),mode="bilinear",align_corners=False)[0].permute(1,2,0)
        w=prob[...,None]
        weighted=(grid*w).reshape(GRID*GRID,-1)            # [G*G,256] target-localized
        proto=(grid*w).sum((0,1))/w.sum().clamp_min(1e-6)  # [256]
    return dict(logit=logit, prob=prob, weighted=weighted, proto=proto,
                peak=float(raw.max()), cls=float(cs[best]), text=output)

if __name__=="__main__":
    rank,local_rank,world_size=P._rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank); device_map={"":f"cuda:{local_rank}"}
    else: device_map="cpu"
    outdir=DATASET/OUT_NAME; outdir.mkdir(parents=True, exist_ok=True)
    videos=P.iter_videos(DATASET,"auto")
    shard=[v for i,v in enumerate(videos) if i%world_size==rank]
    if LIMIT>0: shard=shard[:LIMIT]
    print(f"rank={rank}/{world_size} total={len(videos)} shard={len(shard)} grid={GRID} out={outdir}", flush=True)
    global gen
    gen=InstructSAMTargetMaskGenerator(MODEL, source_root=SRC, device_map=device_map)
    summ=outdir/f"precompute_rank{rank:03d}.jsonl"; ok=err=skip=0; t0=time.time()
    for v in shard:
        op=outdir/f"{v.stem}.pt"
        if op.exists() and os.environ.get("SKIP_EXISTING","1")=="1": skip+=1; continue
        try:
            caption=P.load_caption(DATASET,v.stem); query,phrase=query_for(v.stem,caption)
            img=read_first_frame_image(v)
            with tempfile.NamedTemporaryFile(suffix=".png") as tf:
                img.save(tf.name); r=run(tf.name, query)
            if r is None: raise RuntimeError("no pred_masks")
            payload={"where_logit":r["logit"].contiguous(),
                     "where_prob":r["prob"].contiguous(),
                     "target_dense_weighted":(r["weighted"].half().contiguous() if r["weighted"] is not None else None),
                     "target_proto":(r["proto"].float().contiguous() if r["proto"] is not None else None),
                     "grid":GRID,"query":query,"target_phrase":phrase,"caption":caption,
                     "instructsam_text":r["text"],"peak_logit":r["peak"],"slot_cls":r["cls"],
                     "feature_mode":"where_softlogit+target_dense_weighted"}
            tmp=op.with_suffix(f".rank{rank}.tmp"); torch.save(payload,tmp); os.replace(tmp,op)
            with open(summ,"a") as f: f.write(json.dumps({"status":"ok","stem":v.stem,"peak_logit":r["peak"],"slot_cls":r["cls"],
                "has_dense":r["weighted"] is not None})+"\n")
            ok+=1
        except Exception as e:
            err+=1
            with open(summ,"a") as f: f.write(json.dumps({"status":"error","stem":v.stem,"error":repr(e)})+"\n")
            print(f"[rank{rank}] ERR {v.stem}: {e}", file=sys.stderr, flush=True)
        if (ok+err)%25==0: print(f"rank={rank} ok={ok} skip={skip} err={err} rate={ok/max(time.time()-t0,1e-6):.3f}/s", flush=True)
    print(f"rank={rank} DONE ok={ok} skip={skip} err={err}", flush=True)
