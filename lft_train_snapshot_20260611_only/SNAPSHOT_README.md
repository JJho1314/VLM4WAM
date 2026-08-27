# VLM4WAM

Workspace for staging a fusion of Cosmos Predict 2.5 and InstructSAM, with
`/data/LFT-W02_data/junjie/VLA_WM/Omni-Video` as a reference project.

Imported reference code:

- `third_party/cosmos-predict2.5`
  - Source: `/data/LFT-W02_data/junjie/cosmos-predict2.5`
  - Excluded: `.git`, `.venv`, `outputs`, caches, and logs.
- `third_party/InstructSAM`
  - Source: `/data/LFT-W02_data/junjie/InstructSAM`
  - Excluded: `.git`, caches, checkpoints, logs, work dirs, and generated
    visualization folders.

## Target-Aware Integration

The Cosmos copy now has an InstructSAM-to-Cosmos bridge for target-aware robot
video generation:

- InstructSAM segments the target object from the conditioning image/video first
  frame using `target_query`.
- Cosmos receives InstructSAM's target embedding as cross-attention context:
  `seg_output_embeddings -> mask_hidden_fcs -> TargetFeatureContextAdapter`.
  This follows Omni-Video's VLM-feature path (`norm/proj` into the diffusion
  text/context stream) and uses the same ordering pattern:
  `InstructSAM feature -> Text -> optional mask tokens`.
- The mask is still returned for optional target-attention supervision or
  visualization, but the implicit config does not concatenate it to the
  latent/video input.
- The old explicit mask-channel path is disabled by default:
  `target_mask_concat_input=False` in the DiT and `concat_target_mask=False`
  in the RoboInter training configs.
- TAViD-style target-awareness loss is preserved by supervising selected
  V2T cross-attention blocks (`tavid_attn_alignment_blocks`). The InstructSAM
  feature config uses `tavid_attn_alignment_token_source="text_feature"`, so
  the selective loss can align both the `[TGT]` text token and the prepended
  InstructSAM target feature tokens to the target mask.

Implicit training config:

```bash
experiment=predict2_video2world_training_2b_droid_success_v21_instructsam_implicit_mask
# alias:
experiment=predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context
```

Inference JSON fields:

```json
{
  "name": "target_aware_demo",
  "inference_type": "video2world",
  "input_path": "/path/to/input.mp4",
  "prompt": "A robot arm picks up the target object.",
  "target_query": "Please segment 'the cup' in the image.",
  "instructsam_model_path": "/data/LFT-W02_data/junjie/InstructSAM/work_dirs/stage2",
  "instructsam_feature_mode": "mask_query",
  "target_mask_combine_mode": "best",
  "target_mask_threshold": 0.0
}
```
