import os, torch
import torch.distributed.checkpoint as dcp
M=os.environ["CKPT"]+"/model"
keys=[f"net.blocks.{i}.target_cross_attn_gate" for i in range(28)]
keys+=["net.target_feature_context_adapter.context_gate",
       "net.target_where_prior_head.logit_head.bias",
       "net_ema.target_feature_context_adapter.context_gate"]
keys+=[f"net_ema.blocks.{i}.target_cross_attn_gate" for i in range(28)]
sd={k: torch.zeros(1, dtype=torch.float32) for k in keys}
dcp.load(sd, checkpoint_id=M)
net_g=torch.stack([sd[f"net.blocks.{i}.target_cross_attn_gate"].float().view(()) for i in range(28)])
ema_g=torch.stack([sd[f"net_ema.blocks.{i}.target_cross_attn_gate"].float().view(()) for i in range(28)])
print("=== net.blocks[*].target_cross_attn_gate (init=0.0) @ iter400 ===")
print("  values:", [round(float(v),5) for v in net_g])
print(f"  abs: max={net_g.abs().max():.6f} mean={net_g.abs().mean():.6f} nonzero={(net_g.abs()>1e-6).sum().item()}/28")
print("=== net_ema gate ===")
print(f"  abs: max={ema_g.abs().max():.6f} mean={ema_g.abs().mean():.6f}")
print("context_gate (adapter, init=0.0):", round(float(sd['net.target_feature_context_adapter.context_gate'].view(())),6))
print("prior logit_head.bias (init=-2.0):", round(float(sd['net.target_where_prior_head.logit_head.bias'].view(())),6))
