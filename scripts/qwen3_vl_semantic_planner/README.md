# Qwen3-VL semantic planner

Trains a Qwen3-VL model to act as a **semantic planner**: from the first video frame + instruction it
predicts a dense per-keyframe "semantic plan" that conditions the Cosmos world model
(`data_batch["semantic_plan"]` → cross-attention in the Cosmos DiT). Three independent lines live here;
each is self-contained and does **not** modify the others.

## Lines

### 1. CoVT · SigLIP · 2B  (production baseline)
CoVT-style latent bottleneck: the LM emits a few distinct `<|sem_plan_i|>` latents per keyframe, and
`CoVTLatentDecoderHead` reconstructs the dense SigLIP2 grid ([B, 5·729, 1152]).
- `train_qwen3vl_semantic_planner.py` — trainer (heads: `mlp`, `baton_crossattn`, `covt`)
- `qwen3vl_wrapper.py`, `build_siglip2_semantic_plan_labels.py` — shared model + SigLIP2 target infra
- `sbatch_train_qwen3vl2b_semantic_planner_*.sh` — launchers (base full-FT launcher takes an overridable
  `TRAIN_SCRIPT` so variants reuse it without edits)

### 2. tasktoken · SigLIP · 2B  (variant)
lingbot-style **rich-KV** head: a learnable per-grid query bank cross-attends to
`[ LM latents ⊕ the LLM's own image tokens ]` instead of the CoVT 20-latent bottleneck — targets the CoVT
head's weak per-token identity (low retrieval / small norm). Same SigLIP target ⇒ current WM unchanged.
- `train_qwen3vl_tasktoken_planner.py` (default `--plan-head-type tasktoken`)
- `sbatch_train_qwen3vl2b_tasktoken_online_uniform.sh`

Kept flat (not in a subdir) because it reuses the sibling shared infra (`qwen3vl_wrapper`, SigLIP builder).

### 3. lingbot-DINO · 4B  → `lingbot_dino_4b/`  (new, in progress)
Full lingbot-vla-v2 recipe: base = Qwen3-VL-4B **extracted from the open `robbyant/lingbot-vla-v2-6b`**;
target = **DINO-video** (1024-d), not SigLIP; head = faithful `TaskTokenResampler` warm-started from
lingbot's `future_video_align_head`; loss = plain MSE. Plan = 5 keyframes × 256 tokens = [B, 1280, 1024].
> Switching the target off SigLIP means the **Cosmos WM must be retrained** to consume DINO-video
> conditioning — that is a separate downstream step (block E), not covered by these scripts.

See `lingbot_dino_4b/LINGBOT_DINO_SPEC.md` for the full spec + the training-script swap plan.

## Stage-2: planner ⊗ Cosmos-WM (DIAL-style)
- `planner_plan_provider.py` — load a trained planner ckpt, predict the plan in the WM's `semantic_plan` space
- WM-side hook: `third_party/.../video2world_model_rectified_flow.py` (env-guarded, zero effect unless
  `SEMANTIC_PLAN_PLANNER_CKPT` is set)
- `STAGE2_PLAN.md` — design doc

## Layout
```
qwen3_vl_semantic_planner/
├── train_qwen3vl_semantic_planner.py        # line 1 (CoVT/SigLIP 2B, baseline)
├── train_qwen3vl_tasktoken_planner.py       # line 2 (tasktoken/SigLIP 2B)
├── qwen3vl_wrapper.py, build_siglip2_*.py    # shared infra
├── planner_plan_provider.py, STAGE2_PLAN.md  # stage-2 planner⊗WM
├── sbatch_train_qwen3vl2b_*.sh               # launchers
└── lingbot_dino_4b/                          # line 3 (lingbot-DINO 4B)
    ├── lingbot_resampler.py                  # verbatim TaskTokenResampler port (warm-start key-compat)
    ├── lingbot_dino_head.py                  # 5-keyframe head [B,1280,1024]
    ├── dino_video_target.py                  # DINO-video teacher target encoder
    ├── extract_qwenvl_from_lingbot.py        # extract stock Qwen3-VL-4B from the 6b ckpt
    └── LINGBOT_DINO_SPEC.md                   # spec + swap plan
```
