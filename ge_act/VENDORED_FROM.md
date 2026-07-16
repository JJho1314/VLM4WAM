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

## LTX + SigLIP2 semantic guidance

The LIBERO/LIBERO-Plus data and evaluation entrypoints are preserved from the
source tree. Semantic guidance is an additive extension in
`models/ltx_models/semantic_conditioning.py`; it does not change dataset frame
sampling, camera order, action normalization, or the baseline configs.

The semantic training entrypoint is:

```bash
scripts/train_ltx_siglip2.sh
```

Runtime assets are intentionally not vendored:

- LTX components: `/data/user/jhe724/junjie/weights/LTX-Video`
- GE-Act base: `/data/user/jhe724/junjie/weights/Genie-Envisioner/GE_base_fast_v0.1.safetensors`
- SigLIP2: `/data/user/jhe724/junjie/weights/siglip2-large-patch16-256`
- LIBERO FastWAM data: `/data/user/jhe724/junjie/datasets/LIBERO-fastwam`

Run `scripts/preflight_ltx_siglip2.py` before training. The committed recipe is
8 GPUs, batch 2/GPU, accumulation 8 (global batch 128), bf16, gradient
checkpointing, and 30,000 optimizer steps.
