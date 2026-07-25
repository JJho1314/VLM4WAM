# VLM Planner Attention Heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate paper-ready, dual-camera K4 heatmaps from the VLM planner semantic resampler's real query-to-image attention.

**Architecture:** A standalone visualization script loads the existing exported planner and GE-Act LIBERO dataset, registers a temporary hook on the future semantic head's Perceiver attention, and reconstructs its trained softmax weights without changing model code. Pure helper functions validate and reduce the captured attention, while a renderer writes raw maps, RGB overlays, a MaskWAM-style composite, and a machine-readable manifest.

**Tech Stack:** Python 3.10, PyTorch, Transformers Qwen3-VL, NumPy, Pillow, Matplotlib, pytest.

## Global Constraints

- Support only exported `lingbot_dino`, two-camera, K4 checkpoints with offsets `[2, 4, 6, 8]`.
- Use the future semantic `plan_head`; do not use the depth head, Grad-CAM, or feature similarity.
- Reproduce the trained Perceiver normalization, projections, head reshape, double-square-root scaling, and softmax.
- Remove semantic-latent key columns before forming a spatial heatmap.
- Never combine main- and wrist-camera image tokens in one attention map.
- Default reduction is mean over heads and output-grid queries; `max` is optional and must be recorded.
- Derive the spatial grid from `image_grid_thw` and reject inconsistent token geometry.
- Do not modify model source, checkpoint files, training processes, or dataset files.

---

### Task 1: Attention reconstruction and capture

**Files:**
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_vlm_planner_attention_dual_camera_k4.py`
- Create: `tests/test_vlm_planner_attention_heatmap.py`

**Interfaces:**
- Consumes: `PerceiverAttention` modules whose forward inputs are `(x, latents)`.
- Produces: `reconstruct_perceiver_attention(module, x, latents) -> torch.Tensor`, `reduce_image_attention(weights, image_token_count, reduction) -> torch.Tensor`, and `PlannerAttentionCapture`.

- [ ] **Step 1: Write failing attention-reconstruction tests**

```python
def test_reconstructed_attention_reproduces_perceiver_output():
    module = PerceiverAttention(dim=16, dim_head=4, heads=2).eval()
    x = torch.randn(2, 9, 16)
    latents = torch.randn(2, 5, 16)
    weights, values = reconstruct_perceiver_attention(module, x, latents)
    manual = weights @ values
    manual = manual.permute(0, 2, 1, 3).reshape(2, 5, -1)
    manual = module.to_out(manual)
    torch.testing.assert_close(manual, module(x, latents))


def test_reduce_image_attention_excludes_latent_columns():
    weights = torch.zeros(1, 2, 3, 7)
    weights[..., :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weights[..., 4:] = 1000.0
    reduced = reduce_image_attention(weights, image_token_count=4, reduction="mean")
    torch.testing.assert_close(reduced, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
pytest -q tests/test_vlm_planner_attention_heatmap.py -k "reconstructed or excludes"
```

Expected: collection/import failure because the new script functions do not exist.

- [ ] **Step 3: Implement exact attention reconstruction**

Implement:

```python
def reconstruct_perceiver_attention(module, x, latents):
    x_norm = module.norm1(x)
    latent_norm = module.norm2(latents)
    query = reshape_tensor(module.to_q(latent_norm), module.heads)
    key, value = module.to_kv(torch.cat((x_norm, latent_norm), dim=-2)).chunk(2, dim=-1)
    key = reshape_tensor(key, module.heads)
    value = reshape_tensor(value, module.heads)
    scale = 1.0 / math.sqrt(math.sqrt(module.dim_head))
    weights = torch.softmax(
        ((query * scale) @ (key * scale).transpose(-2, -1)).float(),
        dim=-1,
    ).to(query.dtype)
    return weights, value
```

Implement `reduce_image_attention` with explicit `mean` and `max` branches, finite-value checks, and output shape `[B, image_token_count]`.

Implement `PlannerAttentionCapture` as a context manager that registers one forward hook on `wrapper.plan_head.resampler.layers[0][0]`, stores detached CPU image-token maps, and always removes the hook in `__exit__`.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run:

```bash
pytest -q tests/test_vlm_planner_attention_heatmap.py -k "reconstructed or excludes or capture"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the extraction unit**

```bash
git add qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_vlm_planner_attention_dual_camera_k4.py tests/test_vlm_planner_attention_heatmap.py
git commit -m "feat(viz): capture planner resampler attention"
```

### Task 2: Token-grid restoration and image products

**Files:**
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_vlm_planner_attention_dual_camera_k4.py`
- Modify: `tests/test_vlm_planner_attention_heatmap.py`

**Interfaces:**
- Consumes: reduced attention `[N_image]`, Qwen `image_grid_thw`, camera RGB arrays, and keyframe offsets.
- Produces: `merged_image_grid`, `normalize_attention_stack`, `attention_products`, and `render_composite`.

- [ ] **Step 1: Write failing geometry and normalization tests**

```python
def test_merged_image_grid_matches_qwen_spatial_merge():
    assert merged_image_grid(torch.tensor([1, 18, 18]), 2, 81) == (9, 9)


def test_merged_image_grid_rejects_token_mismatch():
    with pytest.raises(ValueError, match="81"):
        merged_image_grid(torch.tensor([1, 18, 18]), 2, 80)


def test_joint_normalization_is_finite_and_shared_across_k4():
    maps = torch.stack([torch.arange(81).reshape(9, 9).float() + 10 * k for k in range(4)])
    normalized = normalize_attention_stack(maps, lower_quantile=0.02, upper_quantile=0.98)
    assert normalized.shape == (4, 9, 9)
    assert torch.isfinite(normalized).all()
    assert normalized.min() >= 0 and normalized.max() <= 1
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:

```bash
pytest -q tests/test_vlm_planner_attention_heatmap.py -k "grid or normalization"
```

Expected: failures because the geometry and rendering helpers are absent.

- [ ] **Step 3: Implement map restoration and rendering**

Implement:

```python
def merged_image_grid(image_grid_thw, spatial_merge_size, expected_tokens):
    temporal, height, width = (int(v) for v in image_grid_thw.tolist())
    if temporal != 1 or height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError("unsupported Qwen image grid")
    merged = (height // spatial_merge_size, width // spatial_merge_size)
    if merged[0] * merged[1] != expected_tokens:
        raise ValueError(
            f"Qwen merged grid has {merged[0] * merged[1]} tokens, expected {expected_tokens}"
        )
    return merged
```

`normalize_attention_stack` jointly uses the 2nd/98th percentiles over all four maps for one camera. `attention_products` resizes with bilinear interpolation, applies Matplotlib `turbo`, and returns both an unblended color map and `0.55 * heatmap + 0.45 * rgb`.

`render_composite` writes a 2-by-5 figure with one observation column and four offset columns, equal square panels, rounded row containers, `Main Camera`/`Wrist Camera` labels, and the instruction as a compact subtitle.

- [ ] **Step 4: Run rendering tests and confirm GREEN**

Run:

```bash
pytest -q tests/test_vlm_planner_attention_heatmap.py -k "grid or normalization or render"
```

Expected: all selected tests pass and the temporary PNG has nonzero dimensions.

- [ ] **Step 5: Commit the map and rendering unit**

```bash
git add qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_vlm_planner_attention_dual_camera_k4.py tests/test_vlm_planner_attention_heatmap.py
git commit -m "feat(viz): render dual-camera K4 attention maps"
```

### Task 3: Checkpoint and LIBERO CLI integration

**Files:**
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_vlm_planner_attention_dual_camera_k4.py`
- Modify: `tests/test_vlm_planner_attention_heatmap.py`

**Interfaces:**
- Consumes: `--checkpoint-dir`, `--ge-act-data-config`, `--output-dir`, `--num-samples`, `--seed`, `--device`, `--query-reduction`, and `--overlay-alpha`.
- Produces: complete per-sample PNG/NPZ files and `manifest.json`.

- [ ] **Step 1: Write failing metadata and call-order tests**

```python
def test_validate_checkpoint_contract_accepts_dual_camera_k4():
    validate_checkpoint_contract({
        "plan_head_type": "lingbot_dino",
        "num_camera_views": 2,
        "camera_names": ["main", "wrist"],
        "num_keyframes": 4,
        "future_keyframe_offsets": [2, 4, 6, 8],
    })


def test_reshape_capture_order_is_view_major_keyframe_major():
    captures = [torch.full((9,), float(i)) for i in range(8)]
    grouped = group_attention_captures(captures, num_views=2, num_keyframes=4)
    assert grouped.shape == (2, 4, 9)
    assert grouped[1, 0, 0] == 4
```

- [ ] **Step 2: Run CLI-contract tests and confirm RED**

Run:

```bash
pytest -q tests/test_vlm_planner_attention_heatmap.py -k "checkpoint_contract or capture_order"
```

Expected: failures because contract validation and grouping are absent.

- [ ] **Step 3: Implement the CLI data flow**

Reuse:

- `train_semantic_planner.load_ge_act_dual_camera_planner_dataset`;
- `ge_act_dual_camera.DualCameraPlannerCollator`;
- `PlannerWrapper.from_exported_checkpoint`;
- `move_qwen_inputs_to_device`.

For each sample:

1. prepare two independent camera images and one instruction;
2. derive the two merged Qwen image grids from `model_inputs["image_grid_thw"]`;
3. enter `PlannerAttentionCapture`;
4. call only `wrapper.predict_semantic_plan(**model_inputs)` so the depth head is not run;
5. require exactly eight captures and group them `[view=2, keyframe=4, image_tokens]`;
6. write raw arrays, individual products, the composite, and a manifest record.

The manifest is written atomically at the end and includes:

```json
{
  "checkpoint": ".../step_030000",
  "query_reduction": "mean",
  "normalization_quantiles": [0.02, 0.98],
  "overlay_alpha": 0.55,
  "samples": []
}
```

- [ ] **Step 4: Run the complete unit suite**

Run:

```bash
pytest -q tests/test_vlm_planner_attention_heatmap.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit CLI integration**

```bash
git add qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_vlm_planner_attention_dual_camera_k4.py tests/test_vlm_planner_attention_heatmap.py
git commit -m "feat(viz): add planner attention heatmap CLI"
```

### Task 4: Real-checkpoint smoke validation

**Files:**
- Verify: `qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_vlm_planner_attention_dual_camera_k4.py`
- Generate only: `/data/user/jhe724/junjie/outputs/vlm_planner_attention_step30000/`

**Interfaces:**
- Consumes: the HPC3 step-30000 planner checkpoint and predecoded LIBERO GE-Act data config.
- Produces: one complete sample visualization and raw attention archive.

- [ ] **Step 1: Run one-sample smoke inference**

Run on ACD1-25:

```bash
cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python \
  qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_vlm_planner_attention_dual_camera_k4.py \
  --checkpoint-dir /data/user/jhe724/junjie/vlm4wam_joint_assets/planner_step_030000 \
  --ge-act-data-config ge_act/configs/ltx_model/libero/video_model_libero_frozen_qwen_k4_action_30k_hpc3.yaml \
  --output-dir /data/user/jhe724/junjie/outputs/vlm_planner_attention_step30000 \
  --num-samples 1 \
  --device cuda \
  --query-reduction mean
```

Expected: exit code 0 and one composite, eight raw heatmaps, eight overlays, one NPZ, and `manifest.json`.

- [ ] **Step 2: Validate generated artifacts**

Run:

```bash
test -s /data/user/jhe724/junjie/outputs/vlm_planner_attention_step30000/sample_00_planner_attention.png
python -c "import json; p='/data/user/jhe724/junjie/outputs/vlm_planner_attention_step30000/manifest.json'; d=json.load(open(p)); assert len(d['samples']) == 1"
```

Expected: both commands exit 0.

- [ ] **Step 3: Inspect the composite**

Open `sample_00_planner_attention.png` and verify:

- main camera is the first row and wrist camera is the second;
- the observation and all four overlays are spatially aligned;
- labels are `t+2`, `t+4`, `t+6`, and `t+8`;
- maps are finite and not constant;
- no panel contains tokens from the other camera.

- [ ] **Step 4: Run regression tests**

Run:

```bash
pytest -q \
  tests/test_vlm_planner_attention_heatmap.py \
  tests/test_ge_act_dual_camera_planner.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit any smoke-only correction**

If the smoke run required no correction, skip this commit. Otherwise commit only the script and its test:

```bash
git add qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_vlm_planner_attention_dual_camera_k4.py tests/test_vlm_planner_attention_heatmap.py
git commit -m "fix(viz): align planner attention overlays"
```

