"""LTX load milestone: build the multiview LTX transformer (semantic_plan_context) from the run
config + load the trained step_25000 weights (transformer_blocks + semantic_adapter)."""
import os, sys, json, ast, torch
GE="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/ge_act"
os.chdir(GE); sys.path.insert(0, GE)
from utils.model_utils import load_diffusion_model
from models.ltx_models.transformer_ltx_multiview import LTXVideoTransformer3DModel

D="/data/LFT-W02_data/junjie/ltx_semantic_ckpt"
d=json.load(open(f"{D}/config.json"))
dm=ast.literal_eval(d["diffusion_model"]) if isinstance(d["diffusion_model"],str) else d["diffusion_model"]
cfg=dm["config"]
print("building LTX transformer, semantic blocks:",len(cfg.get("semantic_plan_cross_attention_blocks",[])),flush=True)
model=load_diffusion_model(LTXVideoTransformer3DModel, model_dir=f"{D}/step_25000", load_weights=True, **cfg)
model=model.eval()
nblk=len(model.transformer_blocks)
has_sem=hasattr(model,"semantic_adapter")
sem_blocks=sum(1 for b in model.transformer_blocks if getattr(b,"semantic_cross_attention",False))
print(f"blocks={nblk} semantic_adapter={has_sem} semantic_blocks={sem_blocks}",flush=True)
print("M1-LTX-LOAD-OK",flush=True)
