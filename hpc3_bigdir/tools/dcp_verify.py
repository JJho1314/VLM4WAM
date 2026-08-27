import os, torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
M=os.environ["CKPT"]+"/model"
md=FileSystemReader(M).read_metadata(); sm=md.state_dict_metadata
def shp(k): return tuple(sm[k].size)
def dt(k): return getattr(sm[k],'properties',None) and sm[k].properties.dtype
refs={
 "net.blocks.0.target_cross_attn_gate": None,
 "net.blocks.0.target_cross_attn.k_proj.weight": None,   # reference trained weight
 "net.blocks.0.target_cross_attn.q_norm.weight": None,   # ref norm (init 1.0)
 "net.target_where_prior_head.logit_head.bias": None,
 "net.target_where_prior_head.logit_head.weight": None,
}
for k in refs:
    print(k, "stored_dtype=", dt(k), "shape=", shp(k))
sd={k: torch.zeros(shp(k), dtype=torch.float32) for k in refs}
dcp.load(sd, checkpoint_id=M)
import numpy as np
for k,v in sd.items():
    v=v.float()
    print(f"\n{k}\n  shape={tuple(v.shape)} min={v.min():.5f} max={v.max():.5f} mean={v.mean():.5f} std={v.std():.5f} #uniq={len(torch.unique(v))}")
