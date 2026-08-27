# SG-WAM improvement: Option 2 (auxiliary spatial-grounding loss) + Option 5 (task-level eval)

Motivation (from this repo's rigorous probes): the planner-predicted SigLIP `semantic_plan` carries
semantic CONTENT but is NOT target-localizable (linear probe, 160 scenes, 5-fold CV: SG≈baseline,
p>0.1). SigLIP penultimate DOES encode object position (a FiLM+conv head recovers it via CLIPSeg
distillation), but nothing FORCES the plan to expose it, and the model can shortcut via RGB+text.
Plan-X works because it discretizes SigLIP into text-aligned tokens + LLM planning; MaskWAM works
because masks give a hard spatial anchor + an explicit mask-prediction objective. Option 2 ports that
"predict the target's spatial extent" force into our continuous SigLIP setup as an auxiliary loss —
no discrete/autoregressive tokens, no mask input at inference.

## Option 2 — auxiliary spatial-grounding loss (training-code change; run on HPC3/pod)

Files: `semantic_loc_aux.py` (head + loss), `precompute_target_masks.py` (CLIPSeg supervision).

1. Precompute supervision (offline, once; ge-act env):
   `python precompute_target_masks.py`  -> /data/.../LIBERO-target-masks/{suite}_{ep}.npz
   Each: target_masks[V,K,16,16] (CLIPSeg soft mask of the target noun on each keyframe), target_noun_emb[512].
   ALIGN KFI/camera order with your GEActDualCameraPlannerDataset (future_keyframe_offsets, num_camera_views).

2. Dataset: load {target_masks, target_noun_emb} for each sample; collate into the batch.

3. `joint_vlm_geact.py` — JointVLMGEActModel:
   - __init__:  `from ....semantic_loc_aux import SemanticLocalizationHead, semantic_localization_loss`
                `self.loc_head = SemanticLocalizationHead(plan_dim=1024)`
   - forward, right after `semantic_plan, depth_plan, planner_losses = planner_result`:
       ```
       loc = semantic_localization_loss(self.loc_head, semantic_plan, batch["target_noun_emb"],
                 batch["target_masks"], num_keyframes=self.num_keyframes,
                 tokens_per_keyframe=self.tokens_per_keyframe, mask_valid=batch.get("target_valid"))
       planner_losses = {**planner_losses, "loc_loss": loc}
       ```
     (semantic_plan must keep its graph -> gradients reach the planner/Qwen.)

4. `ge_trainer.py` — combine_joint_training_loss: add a `loc_loss_weight` (e.g. 0.5) and
       `+ float(loc_loss_weight) * planner_losses["loc_loss"]`.
   Add loc_loss_weight to optimizer_group_lrs/config; log loc_loss.

Expected: forces the predicted SigLIP plan to become target-localizable (the probe metric should then
rise for SG vs a no-aux baseline — that IS a fair internal check). Also try feeding the loc-head heatmap
as an extra plan channel (Option 3) if the aux loss alone is not enough.

## Option 5 — task-level validation (the honest measure; do NOT use the ad-hoc probes)

Run LIBERO success with the plan ON vs OFF, especially on language-AMBIGUOUS tasks (like MaskWAM's
Tasks 5-8) where the plan should matter most.

- eval script: `ge_act/experiments/eval_libero_official.py`
    `python eval_libero_official.py --ckpt_path <joint_ckpt> --task_suite_name libero_goal --num_trails_per_task 50`
- plan ON  = planner-provided semantic_plan (default of the joint model / planner mode).
- plan OFF = pass semantic_plan=None in the rollout (add a `--no_plan` flag that sets the LTX
  `semantic_plan=None` in the inference call; one-line toggle in the eval's model call).
- Report success-rate ON vs OFF (paired per task). ON >> OFF on ambiguous tasks = the semantic
  guidance is doing real work; this is the result to put in the paper.

Compute: LIBERO rollouts are sim-heavy (mujoco); run on a box with the LIBERO sim installed, not the
quick local probes. The ad-hoc feature-probe / prediction tests in ../sg_probe/ were confounded and
should NOT be used to claim SG>baseline — see ../sg_probe/RESULT_honest.md.

## Option 6 — discrete text-aligned plan via VQ + parallel code prediction (Qwen3-VL-2B, no autoregression)

File: `sg_discrete_plan.py` (validated: forward/backward, gradients reach the planner query features).
Gets Plan-X's discrete/text-grounded benefit without an autoregressive generator.

Pieces:
  - `SigLIPVQ(num_codes=2048, dim=1024)` — EMA codebook over GT SigLIP2 features. Codes inherit SigLIP's
    text-alignment (cheap TA-Tok surrogate). `quantize(gt)->codes`, `embed(codes)->1024-d vectors`.
  - `ParallelCodePlanHead(query_dim=<Qwen query dim, e.g. 1536>, num_codes=2048)` — per-token code logits.
  - `discrete_plan_loss(logits, gt_siglip, vq)` — CE(predicted code | token) vs GT feature's nearest code.
  - `predict_plan_vectors(logits, vq)` — inference: argmax -> codebook 1024-d plan -> WAM UNCHANGED.

Integrate into qwen3_vl_semantic_planner (train_qwen3vl4b_lingbot_dino_planner.py, use the 2B config):
  1. build once:  `self.vq = SigLIPVQ(2048, 1024); self.code_head = ParallelCodePlanHead(QUERY_DIM, 2048)`
     (QUERY_DIM = the lingbot per-keyframe-token query feature width feeding the current plan head).
  2. training: replace the continuous plan-MSE with
       `logits = self.code_head(query_feat)               # [B,V,K,P,2048]`
       `loss_plan, _ = discrete_plan_loss(logits, gt_siglip_plan, self.vq)`
     Keep depth/other planner losses as-is; swap only the semantic-plan objective.
  3. inference / feeding the WAM:
       `plan = predict_plan_vectors(logits, self.vq)       # [B,V,K,P,1024]`  -> pass as semantic_plan.
     WAM (LTX semantic_adapter, in_dim=1024) needs NO change.

### Option 6 on Qwen3.5-2B  (`sg_qwen35_plan.py`, validated end-to-end)

Wires the discrete-plan pieces onto the new **Qwen3.5-2B** VLM (`Qwen3_5ForConditionalGeneration`,
hidden=2048). Runs in the `qwen35` conda env (transformers 5.14.1 + torch 2.7.1); model at
`/data/LFT-W02_data/junjie/weights/Qwen3.5-2B` (downloaded no-proxy). Launch:
  `env -u http_proxy -u https_proxy HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
     /data/LFT-W02_data/.conda/envs/qwen35/bin/python sg_qwen35_plan.py`

Pipeline (`Qwen35DiscretePlanner`):
  1. Qwen3.5-2B reads (current frame + instruction) -> last hidden states [B, L, 2048] (frozen here; LoRA in real train).
  2. `PlanQueryModule`: K*P learnable queries cross-attend to those hidden states -> [B, K*P, 2048] query feats
     (stand-in for the lingbot query block — swap in the real one for the full planner).
  3. `ParallelCodePlanHead(2048, 2048)` -> per-token code logits [B,V=1,K,P,2048] (parallel, no autoregression).
  4. `discrete_plan_loss` vs GT SigLIP codes via `SigLIPVQ`; `predict_plan_vectors` -> [B,V,K,P,1024] plan -> WAM UNCHANGED.
Validated: code-CE drops 7.65->4.0 after codebook warmup, gradients reach query+code heads, 584/2048 codes used.
Heads run fp32 (Qwen hidden is upcast to fp32); real training uses real keyframe SigLIP as GT (dummy GT here).

Notes:
  - Warm up the codebook: run a few hundred steps of ema_update on GT SigLIP before trusting CE, or
    pre-fit the codebook with k-means on a SigLIP feature dump (more stable than pure EMA-from-random).
  - To make discreteness STRICTER text-aligned: add a small contrastive term pulling each code toward
    the CLIP-text embedding of its dominant noun (optional; codebook already text-aligned via SigLIP).
  - Combine with Option 2: predict codes (6) AND add the target-mask aux loss (2) on `embed(codes)` so the
    discrete plan is both text-grounded and spatially-grounded. This is the strongest non-autoregressive
    version of the SG-WAM idea.
