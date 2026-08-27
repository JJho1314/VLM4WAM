# Wan FastWAM action-to-video attention (MaskWAM-style, local)

MaskWAM (Yu et al 2026, Wan-backbone) visualizes **action-to-video attention** (the action expert's
attention to video tokens in the MoT joint attention) — NOT video->text. Reproduced here on our Wan FastWAM.

- **Model**: official Wan FastWAM `wanfastwam_fastwam_official_v3par3/checkpoints/pytorch_model.pt`
  (MoT: mixtures.video + mixtures.action, trained action expert). Built via `create_fastwam` with the
  Wan2.2-TI2V-5B base (VAE/text weights symlinked from weights/DiffSynth-Studio to avoid proxy download).
- **Hook**: MoT `_mixed_attention` (fastwam/models/wan22/mot.py) — action queries attend to
  [k_video(Sv=first-frame 7x7=49) ; k_action]. action->video = softmax(q_action @ k_all)[:, :Sv],
  mean over heads+action-queries, averaged over all blocks. Triggered via `model.infer` (KV-cache path).
- **Env**: `fastwam` conda env; run on GPU1 (shared box OOM-kills GPU0). HF offline.
- `m1_wan_load.py` load milestone; `wan_action_video.py` -> ../figs/wan_action_video.png.
- **Result**: our Wan FastWAM (= the RGB/text baseline in the MaskWAM paper) attends to the arm/gripper
  + workspace, NOT sharply the target object. Consistent with MaskWAM's claim that mask supervision is
  what yields precise target focus; the text/RGB baseline is diffuse. Latent is 7x7 (Wan 32x eff. compression).

## SG-WAM (joint VLM+GE-Act, our model) action-to-video — baseline vs semantic-guided
The REAL SG-WAM WITH action expert IS trained: `weights/joint_vlm_geact_action_k4_50k/step_40000/`
(ltx/ has transformer_blocks + action_blocks(711 keys) + semantic_adapter(346); planner/; joint_meta.json,
train_mode "all", 40k steps). `sg_action_video.py` (ge-act env) loads it, hooks action_blocks[i].attn2
(action queries -> video features = action->video attention), runs WITH semantic_plan (SG) vs None (baseline),
2 rows -> ../figs/sg_action_video.png. action_states/history are 14-dim (action_proj_in in=14),
action_timestep 2D [B,AH]. Note: uses RANDOM action tokens -> attention is arm/workspace-dominated and the
SG-vs-baseline delta is subtle; real denoised actions + multi-step would sharpen it. 8x8 latent (LTX 32x).

## SG-WAM action->video with REAL actions (updated)
`sg_action_video.py` now feeds REAL normalized LIBERO actions (not random): reads action[7]/state[8]
from lerobot parquet, q01/q99 min-max normalize to [-1,1] (libero_fastwam_mix.json per-suite delta_eef/
state_eef), pad both to 14-dim, light noise SSA=0.1. Sink outliers (top-2 tokens) clipped in overlay.
-> ../figs/sg_action_video.png (2 rows: baseline no-plan / SG semantic-plan). Attention is now task-driven
(arm + table objects + workspace). SG-vs-baseline delta is present but MODEST (SG focuses a bit more on the
target objects in some scenes) — semantic plan affects action->video only indirectly (plan -> video feats ->
action attends), so at 40k steps the effect is gentle, not dramatic. 8x8 latent (LTX 32x).

## 3-row figure (baseline / SG / difference)
sg_action_video.png is now 3 rows: baseline (no plan), SG-WAM (semantic plan), and SG-baseline
"added focus" (nrm(sg)-nrm(base), positive-only) which isolates what the semantic plan adds to the
action's attention. The difference row shows SG shifts action attention toward the target objects /
workspace regions vs baseline — the semantic-guidance contribution, though modest at step 40000.
