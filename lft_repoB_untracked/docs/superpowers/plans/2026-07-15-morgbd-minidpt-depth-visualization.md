# MoRGBD MiniDPT v2 Depth Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a dense MiniDPT visualization probe for the current 4B MoRGBD features and generate reference-style, separate main/wrist current/future figures.

**Architecture:** A frozen MoGe-to-MoRGBD teacher supplies `16x16x1024` token maps and dense log-depth targets. A feature-only MiniDPT decoder learns the dense mapping, then decodes both teacher and frozen planner features; its square composite output is restored to `224x448` and split into two `224x224` cameras.

**Tech Stack:** Python, PyTorch, PIL, Matplotlib, pytest, existing FastWAM/LIBERO dataset and Qwen3-VL planner helpers.

## Global Constraints

- Do not retrain or modify the 4B planner checkpoint.
- Do not feed RGB pixels or target depth into planner-feature decoding.
- Use the frozen MoGe dense output as the MiniDPT supervision and reference.
- Keep generated probes, caches, images, temporary tests, and process documents out of the GitHub code-only commit.
- Validate on deterministic windows disjoint from the probe-training cache.

---

### Task 1: Dense MiniDPT Probe and Losses

**Files:**
- Create: `scripts/qwen3_vl_semantic_planner/morgbd_minidpt_probe.py`
- Test: `tests/test_morgbd_minidpt_probe.py`

**Interfaces:**
- Consumes: token tensor `[B,256,1024]` and dense depth `[B,H,W]`.
- Produces: `MiniDPTDepthProbe.forward(tokens) -> [B,1,224,224]`, `dense_log_depth_target(depth)`, `silog_loss(pred, target)`, and `multiscale_gradient_loss(pred, target)`.

- [ ] **Step 1: Write failing architecture and loss tests**

```python
def test_minidpt_outputs_dense_log_depth():
    probe = MiniDPTDepthProbe(in_dim=32, feat=32, grid=4, output_size=28)
    output = probe(torch.randn(2, 16, 32))
    assert output.shape == (2, 1, 28, 28)
    assert torch.isfinite(output).all()

def test_dense_target_and_losses_are_scale_invariant():
    depth = torch.linspace(1, 4, 64).reshape(1, 8, 8)
    target = dense_log_depth_target(depth, output_size=8)
    shifted = target + 2.0
    assert silog_loss(shifted, target) < 1.0
    assert multiscale_gradient_loss(shifted, target) == pytest.approx(0.0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_morgbd_minidpt_probe.py`

Expected: collection fails because `morgbd_minidpt_probe` does not exist.

- [ ] **Step 3: Implement the feature-only MiniDPT**

Implement a 1x1 projection, `8/16/32/64` reassembly branches, residual refinement blocks, coarse-to-fine fusion, and a one-channel log-depth head. Validate token count and feature dimension before reshaping. Port SILog and multi-scale gradient loss from the supplied DA3 v2 reference, with no image input in the probe interface.

- [ ] **Step 4: Run Task 1 tests and verify GREEN**

Run: `pytest -q tests/test_morgbd_minidpt_probe.py`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1 production code**

```bash
git add scripts/qwen3_vl_semantic_planner/morgbd_minidpt_probe.py
git commit -m "feat: add MoRGBD MiniDPT depth probe"
```

Do not add the temporary pytest file.

### Task 2: Probe Cache, Training, and Validation

**Files:**
- Create: `scripts/qwen3_vl_semantic_planner/train_morgbd_minidpt_probe.py`
- Test: `tests/test_train_morgbd_minidpt_probe.py`

**Interfaces:**
- Consumes: four LIBERO suite datasets, frozen `DepthTargetEncoder`, deterministic train/eval indices, and Task 1 probe/loss functions.
- Produces: `minidpt_depth_probe.pt`, `training_history.json`, `validation_metrics.json`, and an optional reusable `teacher_cache.pt` under the requested artifact directory.

- [ ] **Step 1: Write failing cache and validation tests**

Test that cache records contain features `[N,256,1024]` and dense targets `[N,1,224,224]`; train and validation index sets are disjoint; checkpoint configuration can reconstruct the exact probe class; and validation chooses the lowest-loss state rather than the last state.

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_train_morgbd_minidpt_probe.py`

Expected: collection fails because the training module does not exist.

- [ ] **Step 3: Implement teacher-cache extraction**

Reuse the existing FastWAM dataset, disjoint-index selection, and `extract_depth_teacher_outputs` path so inputs exactly match planner training. Cache current and future MoRGBD features in bfloat16 and resized dense log-depth targets in float16 on CPU. Record suite, dataset index, and timepoint for every frame.

- [ ] **Step 4: Implement deterministic MiniDPT training**

Use AdamW, cosine decay, validation every fixed interval, mixed precision, and the objective:

```python
loss = silog_loss(pred_log_depth, target_log_depth) + 0.5 * multiscale_gradient_loss(
    pred_log_depth,
    target_log_depth,
)
```

Save only the best validation state with its architecture, optimizer-independent metadata, split seed, sample counts, and teacher configuration.

- [ ] **Step 5: Implement structural validation metrics**

Report scale-aligned AbsRel, RMSE, delta-1, log-depth Pearson correlation, and multi-scale gradient error for the teacher MiniDPT decode. Load the existing linear probe and report the same metrics as a baseline on identical windows.

- [ ] **Step 6: Run Task 1 and Task 2 tests**

Run: `pytest -q tests/test_morgbd_minidpt_probe.py tests/test_train_morgbd_minidpt_probe.py`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2 production code**

```bash
git add scripts/qwen3_vl_semantic_planner/train_morgbd_minidpt_probe.py
git commit -m "feat: train dense MoRGBD depth probes"
```

### Task 3: Reference-Style Dual-Camera Visualization

**Files:**
- Create: `scripts/qwen3_vl_semantic_planner/visualize_morgbd_minidpt_probe.py`
- Test: `tests/test_morgbd_minidpt_visualization.py`

**Interfaces:**
- Consumes: frozen planner checkpoint, saved MiniDPT probe, DINO PCA probe, FastWAM items, teacher/planner current/future feature dictionaries.
- Produces: `sample_XX_main.png`, `sample_XX_wrist.png`, individual `224x224` panels, `summary.json`, and `summary.csv`.

- [ ] **Step 1: Write failing geometry and output-contract tests**

```python
def test_unsquish_then_split_preserves_camera_order():
    composite = torch.zeros(1, 1, 224, 224)
    composite[..., :112] = 1
    cameras = unsquish_and_split(composite)
    assert cameras["main"].mean() == pytest.approx(1.0)
    assert cameras["wrist"].mean() == pytest.approx(0.0)

def test_reference_layout_saves_two_camera_figures(tmp_path):
    paths = save_reference_style_sample(output_dir=tmp_path, sample_index=0, **inputs)
    assert {p.name for p in paths if p.name.startswith("sample_")} == {
        "sample_00_main.png",
        "sample_00_wrist.png",
    }
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_morgbd_minidpt_visualization.py`

Expected: collection fails because the visualization module does not exist.

- [ ] **Step 3: Implement dense decode and camera geometry**

Decode a complete `224x224` composite before resizing it to `224x448` and splitting at column 224. Normalize each teacher/planner disparity pair jointly, use the same Turbo colormap bounds, and keep MoGe-full references explicitly separate.

- [ ] **Step 4: Implement the supplied-reference layout**

Write separate main/wrist three-row by six-column figures with RGB, DINO teacher/planner, MiniDPT teacher/planner, and MoGe-full reference panels. Also save every underlying panel as an individual `224x224` PNG.

- [ ] **Step 5: Implement planner evaluation and summaries**

Reuse the current planner-loading and prediction helpers. Decode planner depth strictly from `plans["current_depth"]` and `plans["future_depth"]`. Aggregate the structural metrics from Task 2 by suite, camera, and current/future timepoint.

- [ ] **Step 6: Run all focused tests and static checks**

Run:

```bash
python -m py_compile \
  scripts/qwen3_vl_semantic_planner/morgbd_minidpt_probe.py \
  scripts/qwen3_vl_semantic_planner/train_morgbd_minidpt_probe.py \
  scripts/qwen3_vl_semantic_planner/visualize_morgbd_minidpt_probe.py
pytest -q \
  tests/test_morgbd_minidpt_probe.py \
  tests/test_train_morgbd_minidpt_probe.py \
  tests/test_morgbd_minidpt_visualization.py \
  tests/test_dual_camera_probe_visualization.py
git diff --check
```

Expected: compilation succeeds and all focused tests pass.

- [ ] **Step 7: Commit Task 3 production code**

```bash
git add scripts/qwen3_vl_semantic_planner/visualize_morgbd_minidpt_probe.py
git commit -m "feat: visualize MiniDPT depth by camera"
```

### Task 4: Remote Training, Evaluation, and Code-Only Delivery

**Files:**
- No committed artifact files.
- Remote artifacts: probe checkpoint, cache, logs, metrics, and PNG outputs.
- Local artifacts: `artifacts/morgbd_minidpt_depth_v2_20260715/`.

**Interfaces:**
- Consumes: Tasks 1-3 production scripts and the existing 30K planner checkpoint on Pod `182.242.159.145:30282`.
- Produces: a validated probe, full evaluation, inspected figures, and a production-code-only Git commit.

- [ ] **Step 1: Synchronize only production scripts and run a smoke train**

Run a short cache/train/eval cycle on one suite and verify tensor shapes, finite loss, GPU memory, sample layout, and no RGB access in the planner decode path.

- [ ] **Step 2: Train the full probe on the Pod**

Use the largest stable H100 batch, the approved four-suite train split, 5,000 reference-equivalent optimization steps, AdamW, cosine decay, and validation checkpoint selection. Keep the teacher cache for repeatable reruns.

- [ ] **Step 3: Compare MiniDPT against the linear baseline**

Require improved teacher log-depth correlation and gradient error on the same disjoint windows. If it does not improve, stop and report the representation bottleneck rather than generating a misleading final visualization.

- [ ] **Step 4: Generate and inspect full main/wrist figures**

Run all evaluation windows, inspect representative current/future figures from every suite, and verify camera direction, object boundary alignment, shared teacher/planner color scale, and exact `224x224` individual panel dimensions.

- [ ] **Step 5: Pull artifacts back locally and verify checksums**

Copy the complete result directory into `artifacts/morgbd_minidpt_depth_v2_20260715/`, compare remote/local summary and probe SHA-256 values, and validate image/file counts programmatically.

- [ ] **Step 6: Rewrite local process commits into a code-only commit and push**

Preserve a local backup ref, remove design/plan documents from the pushed history, exclude tests and artifacts, rerun focused verification, and push only the three production scripts to `lingbot-zero2-q64-k1`.
