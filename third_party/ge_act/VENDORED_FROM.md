# Vendored GE-Act (Genie-Envisioner-V1)

Source: /data/LFT-W02_data/junjie/VLA_WM/Genie-Envisioner-V1 (fork JJho1314/GE-act, AgiBot Genie-Envisioner)
Vendored subtree: models/ utils/ data/ configs/ runner/ main.py  (code only; weights/datasets NOT copied)

GE uses absolute imports rooted here (`from models...`, `from utils...`, `from data...`),
so put THIS dir on sys.path (see scripts helper / GE_ACT_ROOT) before importing.

External assets needed at runtime (NOT vendored):
- DiT ckpt: Genie-Envisioner-V1/local_ckpts/libero_cosmos_action_step10000/
- Cosmos base (T5+Wan VAE): /data/LFT-W02_data/junjie/weights/Cosmos-Predict2-2B-Video2World/{vae,text_encoder,tokenizer}
- Env: conda env `ge-act` (starVLA superset; diffusers 0.35.2 OK, no 0.32 downgrade)
- Scheduler gotcha: use_karras_sigmas=True (else divide-by-zero NaN in infer)
