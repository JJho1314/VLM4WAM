# Frame 80 Action-to-Video Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate real plan-off/plan-on SG-WAM action-to-video attention artifacts for LIBERO episode 288 frame 80 beside the source RGB image.

**Architecture:** Add one reusable single-frame exporter beside the existing `sg_action_video.py`. It reads a temporally correct `[77..89]` dual-camera window, uses frame 80's real normalized action/state, hooks every `action_blocks[i].attn2`, and runs paired forwards whose only difference is semantic-plan presence. Pure NumPy helpers are tested locally; the full checkpoint run executes on idle GPU 1 and exports raw maps plus reproducible visualizations.

**Tech Stack:** Python, PyTorch, Diffusers/LTX, SigLIP2, PyArrow/Pandas, PyAV, NumPy, Pillow, Matplotlib, pytest.

## Global Constraints

- Dataset is `libero_10_no_noops_lerobot`, episode `288`, frame `80`.
- Input RGB is `outputs/libero_episode_000288_siglip2_da3_stride16_probe/main/frame_000080/rgb.png`.
- Use the trained `joint_vlm_geact_action_k4_50k/step_40000` SG-WAM checkpoint.
- Use frame 80's real action/state and the configured LIBERO normalization statistics.
- Plan-off and plan-on must share all non-plan inputs and the same fixed noise.
- Capture action-query to video-token attention from `action_blocks[*].attn2`; do not use video-to-text attention or Grad-CAM.
- Preserve the source `rgb.png` byte-for-byte.
- Write all results beside the source image.
- Report diffuse real attention honestly; do not post-process it into artificial localization.

---

### Task 1: Single-frame real action-attention exporter

**Files:**
- Create: `semantic_localization/wan_action_attention/single_frame_action_attention.py`
- Create: `tests/test_single_frame_action_attention.py`
- Produce at runtime: `outputs/libero_episode_000288_siglip2_da3_stride16_probe/main/frame_000080/action_attn_*`

**Interfaces:**
- Consumes: checkpoint/config paths, LIBERO episode video/parquet/metadata, frame index `80`, source RGB path, and GPU device.
- Produces: `normalize_q01_q99(values, stats)`, `normalize_map(values)`, `positive_gain(plan_off, plan_on)`, `extract_frame_map(vector, view, time_index, temporal, height, width)`, `aggregate_layer_maps(layer_maps)`, and CLI `main()`.

- [ ] **Step 1: Write failing pure-helper tests**

```python
def test_positive_gain_keeps_only_added_focus():
    off = np.array([[0.0, 1.0], [0.5, 0.5]])
    on = np.array([[0.0, 0.5], [0.5, 1.0]])
    gain = positive_gain(off, on)
    assert gain[0, 1] == 0.0
    assert gain[1, 1] > 0.0


def test_extract_frame_map_selects_requested_view_and_time():
    values = np.arange(2 * 3 * 2 * 2, dtype=np.float32)
    actual = extract_frame_map(values, view=1, time_index=2, temporal=3, height=2, width=2)
    np.testing.assert_array_equal(actual, values.reshape(2, 3, 2, 2)[1, 2])


def test_aggregate_layer_maps_rejects_nonfinite_or_constant_maps():
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_layer_maps({0: np.array([[np.nan, 1.0]])})
    with pytest.raises(ValueError, match="constant"):
        aggregate_layer_maps({0: np.ones((2, 2))})
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run:

```bash
pytest -q tests/test_single_frame_action_attention.py
```

Expected: FAIL because `single_frame_action_attention.py` does not yet exist.

- [ ] **Step 3: Implement pure map, normalization, and pairing helpers**

Implement exact validation:

```python
def normalize_q01_q99(values, stats):
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    return np.clip(2.0 * (values - low) / (high - low + 1e-6) - 1.0, -1.0, 1.0)


def normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("attention map contains non-finite values")
    span = float(values.max() - values.min())
    if span <= 1e-12:
        raise ValueError("attention map is constant")
    return (values - values.min()) / span


def positive_gain(plan_off, plan_on):
    return np.clip(normalize_map(plan_on) - normalize_map(plan_off), 0.0, None)
```

`extract_frame_map` must reshape the flattened vector as `[views, temporal, height, width]` and bounds-check `view` and `time_index`. `aggregate_layer_maps` must validate every layer and return their arithmetic mean.

- [ ] **Step 4: Implement exact episode-window and real-action loading**

Read dual-camera frames `77..89`, resize to `256×256`, and assert that decoded frame 80 matches the provided RGB after the same resize. Use frames `77..80` as memory and `81..89` as future frames. Read action rows `80:112` and state row `80`, normalize with `libero_fastwam_mix.json`, pad both vectors to 14 dimensions, and reuse the same fixed action-noise tensor in both paired forwards.

- [ ] **Step 5: Implement trained-model loading and per-layer attention capture**

Adapt only the established model construction and preprocessing from `semantic_localization/wan_action_attention/sg_action_video.py`. Hook every `action_blocks[i].attn2`, calculate:

```python
probs = (q @ k.transpose(-1, -2) / math.sqrt(head_dim)).softmax(-1)
video_attention = probs.mean(dim=1).mean(dim=1)[0]
```

Store each block separately. Run `plan_off` with both semantic arguments `None`, then `plan_on` with `semantic_plan` and `build_semantic_plan_times(...)`. Keep noisy video latents, real noisy actions, state, text embedding, timesteps, and all random tensors identical.

- [ ] **Step 6: Implement reproducible exports**

Save:

```text
action_attn_plan_off.png
action_attn_plan_on.png
action_attn_sg_gain.png
action_attn_comparison.png
action_attn_layers.png
action_attn_maps.npz
action_attn_metadata.json
```

Use one shared percentile/gamma transform for the two absolute overlays. The comparison is `RGB | plan-off | plan-on | positive SG gain`. The layer sheet contains every captured block for both runs, labeled by block and condition. Metadata records prompt, checkpoint, dataset/episode/frame, temporal window, action/state normalization, seed, hook path, map shapes, and source/output checksums.

- [ ] **Step 7: Run pure-helper tests**

Run:

```bash
pytest -q tests/test_single_frame_action_attention.py
```

Expected: all tests PASS.

- [ ] **Step 8: Run the real checkpoint on GPU 1**

Run:

```bash
env CUDA_VISIBLE_DEVICES=1 \
  MPLCONFIGDIR=/tmp/vlm4wam-frame80-mpl \
  PYTHONPATH=.:ge_act \
  /data/LFT-W02_data/.conda/envs/ge-act/bin/python \
  semantic_localization/wan_action_attention/single_frame_action_attention.py \
  --device cuda
```

Expected: two paired forwards finish, every action block is captured, and all seven artifacts are saved beside `rgb.png`.

- [ ] **Step 9: Validate real artifacts and source immutability**

Run a fresh validation that:

- loads every PNG with Pillow,
- loads every NPZ map and checks finite/non-constant values,
- checks plan-off and plan-on shapes match,
- confirms metadata says all paired non-plan inputs were identical,
- confirms the current `rgb.png` SHA-256 equals its pre-run SHA-256,
- visually inspects `action_attn_comparison.png`.

- [ ] **Step 10: Commit the implementation**

```bash
git add \
  semantic_localization/wan_action_attention/single_frame_action_attention.py \
  tests/test_single_frame_action_attention.py \
  docs/superpowers/plans/2026-07-29-frame80-action-video-attention.md
git commit -m "feat(viz): export paired frame action attention"
```
