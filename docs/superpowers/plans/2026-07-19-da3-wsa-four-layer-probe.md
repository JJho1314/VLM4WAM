# DA3 WSA Four-Layer Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and use a fair four-layer DA3 WSA depth probe for dual-camera K4 planner visualization while preserving the legacy last-layer probe.

**Architecture:** A focused probe module accepts `[B,4,256,2048]`, applies per-token non-affine LayerNorm, maps the four DA3 layers to DPT scales, and decodes 224x224 log depth. The existing trainer gains a `da3_wsa` mode, while a dedicated dual-camera K4 visualizer routes WSA metadata to the new checkpoint and uses the legacy probe only for last-layer metadata.

**Tech Stack:** Python 3.10, PyTorch, Transformers, Matplotlib, pytest, OLA H100.

## Global Constraints

- Train the probe only from frozen real DA3 four-layer teacher features and DA3-full depth targets.
- Default teacher layer order is exactly `11,15,19,23`.
- Preserve every existing probe CLI choice and checkpoint filename.
- Keep target and planner camera/keyframe dimensions explicit until rendering.
- Do not add generated probes, visualizations, caches, or temporary scripts to git.

---

### Task 1: Four-layer WSA probe module

**Files:**
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/wsa_depth_probe.py`
- Create: `tests/test_da3_wsa_probe.py`

**Interfaces:**
- Consumes: `torch.Tensor` shaped `[B,4,256,2048]`.
- Produces: `WSAMultiLayerDPTProbe.forward(tokens) -> [B,1,224,224]` and `config() -> dict`.

- [ ] **Step 1: Write failing shape, validation, normalization-invariance, and save/load tests**

```python
def test_wsa_probe_shape_and_finite():
    probe = WSAMultiLayerDPTProbe(in_dim=32, feat=16, grid=4, output_size=32)
    result = probe(torch.randn(2, 4, 16, 32))
    assert result.shape == (2, 1, 32, 32)
    assert torch.isfinite(result).all()

def test_wsa_probe_rejects_wrong_geometry():
    probe = WSAMultiLayerDPTProbe(in_dim=32, feat=16, grid=4, output_size=32)
    with pytest.raises(ValueError, match="expected tokens"):
        probe(torch.randn(2, 3, 16, 32))

def test_wsa_probe_is_invariant_to_per_token_scale_and_offset():
    probe = WSAMultiLayerDPTProbe(in_dim=32, feat=16, grid=4, output_size=32).eval()
    tokens = torch.randn(2, 4, 16, 32)
    scale = torch.rand(2, 4, 16, 1) + 0.5
    offset = torch.randn(2, 4, 16, 1)
    torch.testing.assert_close(probe(tokens), probe(tokens * scale + offset), atol=2e-5, rtol=2e-5)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest -q tests/test_da3_wsa_probe.py`

Expected: collection fails because `wsa_depth_probe` does not exist.

- [ ] **Step 3: Implement the minimal probe**

```python
class WSAMultiLayerDPTProbe(nn.Module):
    def __init__(self, in_dim=2048, feat=256, grid=16, output_size=224,
                 teacher_layers=(11, 15, 19, 23), out_ch=1): ...
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self._validate(tokens)
        tokens = F.layer_norm(tokens.float(), (self.in_dim,))
        maps = [projection(tokens[:, i].transpose(1, 2).reshape(...))
                for i, projection in enumerate(self.projections)]
        return self.depth_head(self._fuse(maps))
    def config(self) -> dict: ...
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest -q tests/test_da3_wsa_probe.py`

Expected: all probe tests pass.

---

### Task 2: Add `da3_wsa` probe training mode

**Files:**
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/train_feature_probes.py`
- Modify: `tests/test_da3_wsa_probe.py`

**Interfaces:**
- Consumes: four layers from `DepthAnything3TargetEncoder(align_strategy="wsa_multilayer")`.
- Produces: `da3_depth_wsa_probe.pt` containing `state_dict`, `config`, `which`, `teacher_layers`, and `final_loss`.

- [ ] **Step 1: Add a failing parser/checkpoint-contract test**

```python
def test_probe_cli_exposes_da3_wsa_and_stable_checkpoint_name():
    source = Path(TRAINER).read_text()
    assert '"da3_wsa"' in source
    assert '"da3_wsa": "da3_depth_wsa"' in source
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_da3_wsa_probe.py -k cli`

Expected: assertion fails because `da3_wsa` is absent.

- [ ] **Step 3: Implement WSA teacher extraction and checkpoint metadata**

```python
enc = DepthAnything3TargetEncoder(
    process_res=args.da3_process_res,
    align_strategy="wsa_multilayer",
    teacher_layers=(11, 15, 19, 23),
    device=device,
)
probe = WSAMultiLayerDPTProbe(teacher_layers=enc.teacher_layers).to(device)
teacher_feats = lambda fr: enc._patch_tokens(enc._prep(fr)).float()
```

Reuse the current DA3-full log-depth target and `silog_loss + 0.5 * grad_match` objective.

- [ ] **Step 4: Run probe and existing planner tests**

Run: `pytest -q tests/test_da3_wsa_probe.py tests/test_ge_act_dual_camera_planner.py`

Expected: all tests pass.

---

### Task 3: Dual-camera K4 WSA visualization

**Files:**
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/visualize_qwen3vl2b_siglip2_da3_dual_camera_k4.py`
- Modify: `tests/test_da3_wsa_probe.py`
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/README_probes_viz.md`

**Interfaces:**
- Consumes: dual-camera K4 planner checkpoint metadata and either WSA or last-layer probe checkpoint.
- Produces: one 3x6 PNG per sample/camera/keyframe and a JSON manifest recording sample index, camera, offset, feature metrics, and file path.

- [ ] **Step 1: Write failing probe-routing and feature-layout tests**

```python
def test_wsa_metadata_routes_to_four_layer_probe():
    assert probe_kind_for_metadata({"da3_align_strategy": "wsa_multilayer"}) == "wsa"

def test_depth_features_for_probe_preserves_four_layers():
    target = torch.randn(1, 2, 4, 1024, 2048)
    prediction = torch.randn(1, 2, 1024, 4, 2048)
    target_out, pred_out = depth_features_for_probe(target, prediction, 1, slice(256, 512), "wsa")
    assert target_out.shape == pred_out.shape == (1, 4, 256, 2048)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_da3_wsa_probe.py -k 'metadata or features_for_probe'`

Expected: import fails because the visualizer helpers do not exist.

- [ ] **Step 3: Implement metadata routing, checkpoint loading, rendering, and manifest output**

The WSA route passes all four layers to one `WSAMultiLayerDPTProbe`; the legacy route selects layer 23 and uses `MiniDPTProbe`. Reuse the existing SigLIP2 joint PCA, DA3-full GT, turbo coloring, and 3x6 renderer.

- [ ] **Step 4: Run focused and regression tests**

Run: `pytest -q tests/test_da3_wsa_probe.py tests/test_ge_act_dual_camera_planner.py`

Expected: all tests pass.

---

### Task 4: OLA training and qualitative verification

**Files:**
- Runtime artifact only: `/data/users/junjie/probes_2b/da3_depth_wsa_probe.pt`
- Runtime artifact only: `outputs/viz_dual_camera_k4_wsa_step030000/`

**Interfaces:**
- Consumes: committed source, LIBERO frame cache, DA3-LARGE-1.1, step-030000 planner.
- Produces: trained probe, training log, validation metrics, and regenerated PNGs.

- [ ] **Step 1: Sync only tracked source changes to the OLA checkout**

Run a dry-run first and exclude outputs, caches, checkpoints, logs, and unrelated untracked files.

- [ ] **Step 2: Run a one-step smoke training and checkpoint reload**

Expected: finite loss, saved WSA checkpoint, and identical output before/after reload.

- [ ] **Step 3: Train the WSA probe for 5,000 steps**

Run with the existing LIBERO frame cache, bf16 DA3 teacher, batch size selected from a short memory smoke test, AdamW `2e-4`, and cosine decay.

- [ ] **Step 4: Compare teacher-input probe quality**

Report SILog, spatial correlation, and edge/gradient loss on a fixed deterministic validation subset. Do not compare incompatible last-layer and WSA training losses as if they were the same metric.

- [ ] **Step 5: Regenerate and inspect 24 K4 panels**

Generate three deterministic samples, two cameras, and four offsets; verify PNG count, dimensions, finiteness, manifest completeness, and representative main/wrist near/far images.

- [ ] **Step 6: Run final verification and commit source only**

Run: `pytest -q tests/test_da3_wsa_probe.py tests/test_ge_act_dual_camera_planner.py`

Expected: all tests pass; `git status` shows only unrelated pre-existing user files plus intentional source changes before the source commit.
