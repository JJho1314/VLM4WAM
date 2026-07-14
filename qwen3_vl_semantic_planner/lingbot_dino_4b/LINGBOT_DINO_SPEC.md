# lingbot-vla-v2 DINO-video align — spec for the 4B planner port

Source: verified read of `/data/LFT-W02_data/junjie/VLA_WM/lingbot-vla-v2` (file:line cited inline).
All files for this 4B line live in this directory: `qwen3_vl_semantic_planner/lingbot_dino_4b/`.
Target file to build (here): `train_qwen3vl4b_lingbot_dino_planner.py` (independent; 2B CoVT + tasktoken untouched).

## Teacher (target encoder) — block C
- `build_dino_video_teacher` → `DinoVideoTeacher.build()` (dino_video/teacher.py:132,15).
- Backbone = lumos_dinov3 NaViT video ViT; arch from shipped `dino_video/config.yaml`
  (`cfg.dinov3.student.arch`). Inferred **vit_large, embed_dim 1024, patch 16** (must equal head dim_out
  1024 + input 256 ⇒ 16×16=256 tokens). Confirm from the config.yaml that ships in the HF `dino_video/` dir.
- INPUT: video `[B,C,T=3,H=256,W=256]` = `[warmup(=current.clone()), current, future]`, current_index=1
  (`use_warmup_frame:true`, `num_future_frames:1`). RGB→[0,1]→ImageNet mean/std. bf16, effective_fps 1.0.
- OUTPUT: `get_future_feature` → `get_intermediate_layers(n=1, norm=True)` (final **LayerNorm**, NOT L2) →
  return **future frame's patch tokens `[B, 256, 1024]`**, detached. (teacher.py:73-129)
- Weights: `ckpt_path` + `config_path` ship in release `dino_video/` (README:69). Loader strips `backbone.` prefix.

## Head — block B  (TaskTokenResampler, resampler.py:163)
- dims: **dim_in=dim_mid=llm_hidden=2560, dim_out=1024, num_layers=1, num_heads=4, dim_head=32,
  ff_mult=1, num_queries=num_backbone_tokens=256** (depth_head.py:52; robotwin.yaml:78-139).
- keys: `proj_in1`(→queries), `proj_in2`(→x), `layers.0.0.{norm1,norm2,to_q,to_kv,to_out}` (bias=False),
  `layers.0.1.{0=LN,1=Lin,3=Lin}` (FFN), `proj_out`, `norm_out`. PerceiverAttention: queries cross-attend
  to `cat(x, queries)`; scale=1/sqrt(sqrt(dim_head)); softmax fp32.
- forward (video_emb_forward, modeling_lingbot_vla.py:1445):
  - `image_embs = last_hidden[:, 1:65, :]` = first cam **64** image tokens (image_token_size 8 ⇒ 8×8), DETACHED.
  - `task_tokens` = **last 8 hidden states** (num_task_tokens=8) (shared future-depth query).
  - `x = cat(image_embs[64], task_tokens[8]) = [B,72,2560]`.
  - `queries = future_video_align_embs.repeat(B,1,1)` — learnable param **[256, 2560]**.
  - `preds = head(x, queries) = [B,256,1024]`.
- WARM-START: load whole `future_video_align_head.projector.*` (proj_in1/2, layers.0, proj_out, norm_out) +
  the `future_video_align_embs` [256,2560] query param, from the 6b checkpoint. All dims match at 4B.

## Loss — block D
- **Plain MSE** on raw features: `mse_loss(preds.float(), target.float().detach())`, NOT normalized, NOT
  smooth_L1 (smooth_L1/cosine branches disabled in cfg). (modeling_lingbot_vla.py:1552)
- lingbot weights it 0.004 (auxiliary). For US it is the PRIMARY objective ⇒ weight 1.0.

## Temporal structure — DESIGN FORK
- lingbot predicts **ONE** future frame (num_future_frames=1) ⇒ plan = [256,1024]. No multi-keyframe example.
- Our planner is 5-keyframe. Options:
  - (a) match lingbot: single future keyframe [256,1024]. Max fidelity, simplest, loses temporal plan.
  - (b) 5 keyframes: run the (warm-started, shared) head 5× with a per-keyframe query/pos embedding, targets =
    DINO-video at our 5 keyframe times ⇒ plan = [5,256,1024] = 1280 tokens. RECOMMENDED for a video WM.
- This choice also sets the WM conditioning shape (block E): 1024-dim, 256 or 1280 tokens.
- DECIDED (b): 5 keyframes, plan = [B, 1280, 1024]. Head/target already built for this.

## Training-script glue — swap spec (copy PRISTINE train_qwen3vl_semantic_planner.py → train_qwen3vl4b_lingbot_dino_planner.py)
All swaps verified against the existing script's line anchors:
1. Dataset __getitem__ (~L302-323): add `current_image` = frames[0] (H,W,3 uint8) beside `keyframe_images`.
2. Collator (~L339-363): stack + pass `current_image` through in online mode.
3. main() online encoder (L903-920): replace load_siglip2/encode_images with
   `DinoVideoTargetEncoder(ckpt=<dino_video/teacher_*.pth>, config=<dino_video/config.yaml>)`.
4. Training loop (L1062-1073): pop `current_image` too; `target = dino.encode_future_keyframes(cur[B,3,H,W],
   [keyframes[:,k].permute(0,3,1,2) for k])` → [B,1280,1024]; set `semantic_plan_labels`.
5. Head: new plan_head_type "lingbot_dino" → LingbotDinoPlanHead; semantic_dim=1024, target_len=K*256=1280,
   num_latent_per_keyframe=8 (match lingbot's 8 task tokens) ⇒ 5*8=40 distinct <|sem_plan_i|> tokens.
6. PlannerWrapper.predict_semantic_plan: for lingbot_dino, pass image_hidden (collect_image_hidden, DETACH
   per lingbot detach_image_feats) + latent_hidden to the head.
7. Loss (compute_plan_losses, L607): for lingbot_dino use PLAIN MSE only (block D), weight 1.0.
8. Warm-start: base VLM from extracted 4B (--model-path); head from 6b align-head via head.load_lingbot_warmstart(
   <slice of qwenvl_with_expert...future_video_align_head.* + future_video_align_embs from the 6b shards>).
9. Distinct tokens: same as covt path (in ("covt","lingbot_dino")).
GATE before writing glue: validate DinoVideoTargetEncoder imports+builds in the training env (lumos_dinov3
flex_attention vs sdpa) once the 6b download (with dino_video/) completes.
