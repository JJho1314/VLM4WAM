## Task 8: Add Checkpoint-Backed Smoke Verification and Run the Full Gate

**Files:**

- Create: `third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py`
- Modify: `tests/test_fastwam_cosmos_semantic_plan.py`
- Modify: `tests/test_dino_depth_plan_provider.py` if smoke-discovered contract coverage is missing

- [ ] **Step 1: Add a smoke script argument-contract test**

Add:

```python
def test_online_smoke_script_requires_real_checkpoint_and_config():
    smoke_path = (
        ROOT
        / 'third_party/FastWAM/scripts'
        / 'smoke_online_dino_depth_semantic_plan.py'
    )
    source = smoke_path.read_text()
    assert '--planner-checkpoint' in source
    assert '--config-dir' in source
    assert '--config-name' in source
    assert '--task' in source
    assert 'torch.inference_mode()' in source
```

- [ ] **Step 2: Create the one-batch smoke script**

The script must:

1. Parse `--planner-checkpoint`, `--config-dir`, `--config-name`, `--task`, `--device`, `--instruction`, `--image`, and `--video-fps`. Default `--task` to `libero_cosmos_2cam224_online_dino_depth`.
2. Verify the planner checkpoint with `validate_checkpoint_files` and `validate_planner_metadata` before loading FastWAM.
3. Compose the Hydra config with `overrides=[f'task={args.task}']` and force:

```python
cfg.model.online_semantic_planner = True
cfg.model.online_semantic_planner_checkpoint = str(
    args.planner_checkpoint
)
```

4. Load one RGB image, normalize it to `[-1, 1]`, and call `infer_action` with the raw instruction and explicit sampled-video FPS.
5. Register a temporary forward hook on `video_expert.semantic_plan_fusion` and assert its output shape is exactly `(1, 1024, 1024)`.
6. Assert the action output is finite and has a non-empty temporal dimension.
7. Print only the verified checkpoint path, fused-plan shape, action shape, and sampled FPS.

Use this main structure:

```python
def main():
    args = parse_args()
    validate_checkpoint(args.planner_checkpoint)
    cfg = load_config(args)
    model = create_fastwam_cosmos(cfg).eval()
    image = load_rgb_tensor(args.image, device=args.device)
    captured = {}

    def capture_fused_plan(_module, _inputs, output):
        captured['shape'] = tuple(output.shape)

    handle = model.video_expert.semantic_plan_fusion.register_forward_hook(
        capture_fused_plan
    )
    try:
        with torch.inference_mode():
            result = model.infer_action(
                input_image=image,
                instruction=args.instruction,
                prompt=(
                    "A video recorded from a robot's point of view executing "
                    f'the following instruction: {args.instruction}'
                ),
                video_fps=args.video_fps,
            )
            actions = result['action']
    finally:
        handle.remove()

    if captured.get('shape') != (1, 1024, 1024):
        raise RuntimeError(
            f'unexpected fused plan shape: {captured.get("shape")}'
        )
    if actions.numel() == 0 or not torch.isfinite(actions).all():
        raise RuntimeError('action output is empty or non-finite')
    print(
        {
            'planner_checkpoint': str(args.planner_checkpoint),
            'fused_plan_shape': captured['shape'],
            'action_shape': tuple(actions.shape),
            'video_fps': float(args.video_fps),
        }
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run all CPU-safe unit tests**

Run:

```bash
pytest -q \
  tests/test_lingbot_dino_depth_contract.py \
  tests/test_dino_depth_plan_provider.py \
  tests/test_fastwam_dino_depth_fusion.py \
  tests/test_fastwam_cosmos_semantic_plan.py \
  tests/test_cosmos_semantic_plan_stage2.py
```

Expected: PASS with no skipped test that covers a contract listed in this plan.

- [ ] **Step 4: Run syntax and import checks**

Run:

```bash
python -m compileall -q \
  scripts/qwen3_vl_semantic_planner \
  third_party/FastWAM/src/fastwam/models/cosmos \
  third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py
bash -n \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4.sh
```

Expected: both commands exit with code 0.

- [ ] **Step 5: Train or fine-tune the nine-frame planner**

Run with the actual dataset and output locations:

```bash
FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
FASTWAM_DATASET_DIRS=/data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_spatial_no_noops_lerobot:/data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_object_no_noops_lerobot:/data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_goal_no_noops_lerobot:/data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_10_no_noops_lerobot \
OUTPUT_DIR=/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/qwen3vl4b_dino_depth_fastwam_k4 \
scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4.sh
```

Expected export under the step directory named by `OUTPUT_DIR/latest_checkpoint.txt`:

```text
qwen3vl_lora_or_model/
processor/
plan_head.pt
depth_head.pt
plan_token_embedding.pt
planner_meta.json
```

Resolve the per-step directory and validate its `planner_meta.json`:

```bash
PLANNER_CHECKPOINT=$(tr -d '\n' < \
  /data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/qwen3vl4b_dino_depth_fastwam_k4/latest_checkpoint.txt)
test -d "$PLANNER_CHECKPOINT"
```

The metadata must report:

```json
{
  "sequence_length": 9,
  "num_keyframes": 4,
  "grid_size": 16,
  "semantic_dim": 1024,
  "target_tokens": 1024,
  "keyframe_offsets": [2, 4, 6, 8],
  "shared_latent_per_keyframe": 32,
  "private_latent_per_keyframe": 32,
  "branch_latent_per_keyframe": 64,
  "total_unique_latent_per_keyframe": 96,
  "query_layout": "keyframe_major__shared_dino_private_depth_private",
  "has_depth_head": true,
  "token_order": "keyframe_major_row_major"
}
```

- [ ] **Step 6: Run the checkpoint-backed FastWAM smoke test**

Use an image whose horizontal composition matches the two-camera FastWAM input:

```bash
PLANNER_CHECKPOINT=$(tr -d '\n' < \
  /data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/qwen3vl4b_dino_depth_fastwam_k4/latest_checkpoint.txt)
FASTWAM_PLANNER_CHECKPOINT="$PLANNER_CHECKPOINT" \
python third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py \
  --planner-checkpoint \
  "$PLANNER_CHECKPOINT" \
  --config-dir third_party/FastWAM/configs \
  --config-name train \
  --task libero_cosmos_2cam224_online_dino_depth \
  --device cuda:0 \
  --instruction 'open the middle drawer' \
  --image /absolute/path/to/two_camera_current_frame.png \
  --video-fps 5.0
```

Expected: one printed dictionary with `fused_plan_shape=(1, 1024, 1024)` and a finite, non-empty action shape.

- [ ] **Step 7: Inspect the final diff for scope and accidental placeholders**

Run:

```bash
git status --short
git diff --check
rg -n 'TODO|TBD|NotImplemented|pass$' \
  scripts/qwen3_vl_semantic_planner \
  third_party/FastWAM/src/fastwam/models/cosmos \
  third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py \
  tests/test_lingbot_dino_depth_contract.py \
  tests/test_dino_depth_plan_provider.py \
  tests/test_fastwam_dino_depth_fusion.py \
  tests/test_fastwam_cosmos_semantic_plan.py
```

Expected:

- `git diff --check` produces no output.
- The placeholder scan produces no new placeholder in modified lines.
- `git status --short` contains only intentional work plus pre-existing user changes.

- [ ] **Step 8: Commit the smoke test**

Run:

```bash
git add \
  third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py \
  tests/test_fastwam_cosmos_semantic_plan.py \
  tests/test_dino_depth_plan_provider.py
git commit -m 'test: verify online dino depth fastwam path'
```

---

## Final Acceptance Checklist

- [ ] A production planner checkpoint was trained or fine-tuned with `sequence_length=9` and exports both `plan_head.pt` and `depth_head.pt`.
- [ ] The provider performs one Qwen forward for both branches and returns detached `[B,1024,1024]` DINO and depth tensors.
- [ ] Each keyframe has 32 shared plus 32 private queries per branch; each head receives 64 queries, and the full four-keyframe VLM query sequence contains exactly 384 distinct tokens.
- [ ] FastWAM invokes the provider online in training and inference from the current composed RGB image and raw instruction.
- [ ] The planner remains frozen and is not registered in `FastWAMCosmos.state_dict()`.
- [ ] The fusion module is registered under the Cosmos video expert, receives gradients, and starts with a depth contribution gate of approximately `0.1`.
- [ ] Semantic keyframe times are exactly `[0.25,0.5,0.75,1.0]` for offsets `[2,4,6,8]`.
- [ ] Effective sampled-video FPS is emitted by the dataset and reaches MoT, cross-attention, AGRA standalone video loss, AGRA foresight, and inference.
- [ ] Online and file-backed sources cannot be active together.
- [ ] The full CPU-safe test command, syntax checks, and checkpoint-backed GPU smoke test pass.
