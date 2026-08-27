"""Wan FastWAM load milestone: build MoT (video+action experts) + load official checkpoint (mot)."""
import os, sys, functools, torch
torch.load = functools.partial(torch.load, weights_only=False)
os.environ.setdefault("DIFFSYNTH_MODEL_ROOT", "/data/LFT-W02_data/junjie/weights")
FW="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/FastWAM"
sys.path.insert(0, FW+"/src")
from fastwam.runtime import create_fastwam
CKPT="/data/LFT-W02_data/junjie/eval_runs/wanfastwam_fastwam_official_v3par3/checkpoints/pytorch_model.pt"
dt=torch.bfloat16
DITCFG=dict(patch_size=[1,2,2], in_dim=48, hidden_dim=3072, ffn_dim=14336, freq_dim=256,
            text_dim=4096, out_dim=48, num_heads=24, attn_head_dim=128, num_layers=30, eps=1.0e-06, has_image_input=False)
ADITCFG=dict(action_dim=7, hidden_dim=1024, ffn_dim=4096, freq_dim=256, text_dim=4096,
             num_heads=24, attn_head_dim=128, num_layers=30, eps=1.0e-06)
SCH={"train_shift":5.0,"infer_shift":5.0,"num_train_timesteps":1000}
print("building Wan FastWAM...", flush=True)
model=create_fastwam("Wan-AI/Wan2.2-TI2V-5B", "Wan-AI/Wan2.1-T2V-1.3B", DITCFG,
    load_text_encoder=False, proprio_dim=8, action_dit_config=ADITCFG, action_dit_pretrained_path=None,
    skip_dit_load_from_pretrain=True, video_scheduler=SCH, action_scheduler=SCH, model_dtype=dt, device="cuda")
print("built. loading checkpoint...", flush=True)
ck=torch.load(CKPT, map_location="cpu", weights_only=False)
print("ckpt keys:", list(ck.keys())[:6], flush=True)
# load mot (mixtures video+action) + proprio_encoder
tgt=None
for name,mod in model.named_modules():
    if name.endswith("mixtures") or (hasattr(mod,"video") and hasattr(mod,"action")):
        tgt=mod; print("mixtures module:", name, flush=True); break
sd=ck["mot"]
# strip a leading "mixtures." if present in target's own state_dict naming
res=model.load_state_dict({("mot."+k if not k.startswith("mot.") else k):v for k,v in {}.items()}, strict=False) if False else None
# direct: find the object holding .mixtures and load
sd_str={ (k[len("mixtures."):] if k.startswith("mixtures.") else k):v for k,v in sd.items() }
loaded=False
for name,mod in model.named_modules():
    if hasattr(mod,"mixtures"):
        miss,unexp=mod.mixtures.load_state_dict(sd_str, strict=False)
        miss=[m for m in miss if not m.endswith("_extra_state")]; unexp=[u for u in unexp if not u.endswith("_extra_state")]
        print(f"loaded mot into {name}.mixtures  missing={len(miss)} unexpected={len(unexp)}", flush=True)
        if miss[:4]: print("  miss sample:", miss[:4])
        loaded=True; break
if not loaded: print("WARN: mixtures not found for mot load", flush=True)
if "proprio_encoder" in ck and hasattr(model,"proprio_encoder") and model.proprio_encoder is not None:
    model.proprio_encoder.load_state_dict(ck["proprio_encoder"], strict=False); print("proprio_encoder loaded", flush=True)
model=model.eval()
# locate video DiT blocks
for name,mod in model.named_modules():
    if hasattr(mod,"blocks") and isinstance(getattr(mod,"blocks"),torch.nn.ModuleList) and len(mod.blocks)>=20:
        b0=mod.blocks[0]
        print(f"video DiT: {name} nblocks={len(mod.blocks)} block_has_cross_attn={hasattr(b0,'cross_attn')}", flush=True); break
print("M1-WAN-LOAD-OK", flush=True)
