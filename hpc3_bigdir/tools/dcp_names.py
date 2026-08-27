import os, torch
from torch.distributed.checkpoint import FileSystemReader
M=os.environ["CKPT"]+"/model"
md=FileSystemReader(M).read_metadata()
sd=md.state_dict_metadata
keys=list(sd.keys())
print("total tensors:", len(keys))
import re
for pat in ["target_cross_attn_gate","context_gate","where_gate","logit_head","prior_bias","prior_init","spatial_prior","what_where","target_cross_attn"]:
    hits=[k for k in keys if pat in k]
    if hits:
        print(f"\n# pattern '{pat}': {len(hits)} hits")
        for k in hits[:8]:
            sz=getattr(sd[k],'size',None)
            print(f"   {k}  size={tuple(sz) if sz is not None else '?'}")
