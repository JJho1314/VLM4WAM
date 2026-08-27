# Dual-camera Probe Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-evaluate the saved planner probes and emit spatially aligned, independent 224x224 main-camera and wrist-camera DINO/Depth visualizations.

**Architecture:** Add one focused evaluation entry point that reuses the existing saved DINO PCA and Depth linear probes. It splits the composite 16x16 feature grid into two 16x8 camera grids before interpolation, splits dense MoGe depth before resizing, performs Depth scale alignment per camera, and writes 24 independent PNG files plus auditable color-range metadata per sample. The planner and probes are not retrained.

**Tech Stack:** Python 3.11+, PyTorch, Pillow, Matplotlib, FastWAM LIBERO dataset adapter, Qwen3-VL planner runtime.

## Global Constraints

- Input camera order is exactly `main | wrist`, inherited from `image`, `wrist_image` and `concat_multi_camera: horizontal`.
- Existing planner output remains one 16x16/256-token composite; each camera receives a 16x8/128-token half.
- Split token and dense-depth grids before interpolation; never crop an already colored feature PNG.
- Every output PNG is exactly 224x224.
- For a fixed sample/time/camera, MoGe, teacher-probe, and planner-probe Depth maps use one target-derived 2nd/98th-percentile Viridis range.
- Save no combined-camera image, contact sheet, query box, or query-token frame.
- Reuse `probe224_pca_depth_20260715/dino_pca_probe.pt` and `depth_linear_probe.pt`; do not retrain either probe.
- Design documents, plans, generated artifacts, and tests remain local and are excluded from later GitHub pushes.

---

### Task 1: Camera-aware projection and Depth alignment primitives

**Files:**
- Create: `scripts/qwen3_vl_semantic_planner/visualize_dual_camera_probes.py`
- Test: `tests/test_dual_camera_probe_visualization.py`

**Interfaces:**
- Consumes: `DinoPCAProbe`, `validate_token_features`, `TOKEN_GRID_SIZE=16`, and `OUTPUT_SIZE=224` from `train_dino_depth_probe_visualization.py`.
- Produces: `split_rgb_cameras_224`, `project_dino_cameras_224`, `resize_depth_target_cameras_224`, and `decode_depth_cameras_224`.

- [ ] **Step 1: Write failing camera-split tests**

```python
def test_rgb_split_preserves_main_then_wrist_order():
    module = load_module()
    composite = torch.zeros(224, 448, 3, dtype=torch.uint8)
    composite[:, :224, 0] = 255
    composite[:, 224:, 1] = 255
    cameras = module.split_rgb_cameras_224(composite)
    assert tuple(cameras) == ("main", "wrist")
    assert cameras["main"].getpixel((0, 0)) == (255, 0, 0)
    assert cameras["wrist"].getpixel((0, 0)) == (0, 255, 0)


def test_dino_tokens_are_split_before_interpolation():
    module = load_module()
    mean = torch.zeros(4)
    basis = torch.eye(4)[:, :3]
    probe = module.DinoPCAProbe(mean, basis, torch.zeros(3), torch.ones(3))
    features = torch.zeros(1, 256, 4)
    grid = features.reshape(1, 16, 16, 4)
    grid[:, :, :8, 0] = 1.0
    grid[:, :, 8:, 1] = 1.0
    cameras = module.project_dino_cameras_224(probe, features)
    assert cameras["main"].shape == (1, 3, 224, 224)
    assert cameras["wrist"].shape == (1, 3, 224, 224)
    assert float(cameras["main"][:, 0].mean()) == pytest.approx(1.0)
    assert float(cameras["main"][:, 1].mean()) == pytest.approx(0.0)
    assert float(cameras["wrist"][:, 0].mean()) == pytest.approx(0.0)
    assert float(cameras["wrist"][:, 1].mean()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest -q tests/test_dual_camera_probe_visualization.py \
  -k 'rgb_split or dino_tokens'
```

Expected: FAIL because `visualize_dual_camera_probes.py` and its split functions do not exist.

- [ ] **Step 3: Implement strict RGB and DINO splitting**

```python
CAMERAS = ("main", "wrist")


def _split_width(value: torch.Tensor, *, name: str) -> dict[str, torch.Tensor]:
    if value.shape[-1] % 2:
        raise ValueError(f"{name} width must be even, got {value.shape[-1]}")
    midpoint = value.shape[-1] // 2
    return {"main": value[..., :midpoint], "wrist": value[..., midpoint:]}


def split_rgb_cameras_224(value: Any) -> dict[str, Image.Image]:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.shape != (224, 448, 3):
        raise ValueError(f"RGB composite must be [224,448,3], got {tuple(tensor.shape)}")
    halves = _split_width(tensor.permute(2, 0, 1), name="RGB composite")
    return {
        camera: Image.fromarray(half.permute(1, 2, 0).to(torch.uint8).numpy())
        for camera, half in halves.items()
    }


def project_dino_cameras_224(
    probe: DinoPCAProbe,
    features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_token_features(features, feature_dim=probe.mean.numel(), name="DINO features")
    projected = (features.float() - probe.mean) @ probe.basis
    projected = ((projected - probe.low) / (probe.high - probe.low).clamp_min(1e-6)).clamp(0, 1)
    grid = projected.reshape(-1, 16, 16, 3).permute(0, 3, 1, 2)
    return {
        camera: F.interpolate(half, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        for camera, half in _split_width(grid, name="DINO token grid").items()
    }
```

- [ ] **Step 4: Write failing per-camera Depth alignment tests**

```python
def test_depth_alignment_is_independent_per_camera():
    module = load_module()
    relative = torch.zeros(1, 16, 16)
    dense = torch.ones(1, 256, 256)
    dense[..., 128:] = 10.0
    decoded = module.decode_depth_cameras_224(relative, dense)
    assert decoded["main"].median() == pytest.approx(1.0)
    assert decoded["wrist"].median() == pytest.approx(10.0)


def test_camera_split_rejects_odd_width():
    module = load_module()
    with pytest.raises(ValueError, match="width must be even"):
        module.resize_depth_target_cameras_224(torch.ones(1, 32, 31))
```

- [ ] **Step 5: Run the tests and verify RED**

Run:

```bash
pytest -q tests/test_dual_camera_probe_visualization.py \
  -k 'depth_alignment or odd_width'
```

Expected: FAIL because the Depth camera functions do not exist.

- [ ] **Step 6: Implement per-camera target resizing and log-scale alignment**

```python
def resize_depth_target_cameras_224(target: torch.Tensor) -> dict[str, torch.Tensor]:
    target = _as_bhw(target, name="target_depth").float().clamp_min(1e-6)
    return {
        camera: F.interpolate(
            half.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False
        )[:, 0]
        for camera, half in _split_width(target, name="dense Depth target").items()
    }


def decode_depth_cameras_224(
    relative: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    relative = _as_bhw(relative, name="relative_log_prediction").float()
    target_cameras = resize_depth_target_cameras_224(target)
    result = {}
    for camera, half in _split_width(relative, name="Depth token grid").items():
        prediction = F.interpolate(
            half.unsqueeze(1), size=(224, 224), mode="bicubic", align_corners=False
        )[:, 0]
        prediction -= prediction.mean(dim=(-2, -1), keepdim=True)
        truth = target_cameras[camera].to(prediction.device)
        shift = (truth.log() - prediction).flatten(1).median(dim=1).values[:, None, None]
        result[camera] = (prediction + shift).exp()
    return result
```

- [ ] **Step 7: Run Task 1 tests and commit**

Run:

```bash
pytest -q tests/test_dual_camera_probe_visualization.py
```

Expected: PASS.

Commit only the production script. Keep the test local:

```bash
git add scripts/qwen3_vl_semantic_planner/visualize_dual_camera_probes.py
git commit -m "feat: split planner probes by camera"
```

---

### Task 2: Saved-probe evaluation, fair Depth outputs, and camera metrics

**Files:**
- Modify: `scripts/qwen3_vl_semantic_planner/visualize_dual_camera_probes.py`
- Test: `tests/test_dual_camera_probe_visualization.py`

**Interfaces:**
- Consumes: Task 1 camera projection functions and the existing planner/teacher helpers from `train_dino_depth_probe_visualization.py`.
- Produces: `load_saved_probes`, `save_dual_camera_sample`, CLI `main`, `summary.json`, `summary.csv`, and one `depth_color_ranges.json` per sample.

- [ ] **Step 1: Write failing saved-probe and output-contract tests**

```python
def test_dual_camera_output_contract_has_24_separate_pngs(tmp_path):
    module = load_module()
    dino = {
        f"dino_{source}_{camera}_{time}_224": torch.zeros(3, 224, 224)
        for source in ("teacher", "planner")
        for camera in module.CAMERAS
        for time in ("current", "future")
    }
    depth = {
        f"depth_{source}_{camera}_{time}_224": torch.ones(224, 224)
        for source in ("moge", "teacher_probe", "planner_probe")
        for camera in module.CAMERAS
        for time in ("current", "future")
    }
    module.save_dual_camera_sample(
        output_dir=tmp_path,
        current_rgb=torch.zeros(224, 448, 3, dtype=torch.uint8),
        future_rgb=torch.zeros(224, 448, 3, dtype=torch.uint8),
        instruction="pick up the bowl",
        dino_maps=dino,
        depth_maps=depth,
    )
    pngs=list(tmp_path.glob("*.png"))
    assert len(pngs) == 24
    assert not any("combined" in path.name or "query" in path.name for path in pngs)
    assert (tmp_path / "depth_color_ranges.json").is_file()
    for path in pngs:
        with Image.open(path) as image:
            assert image.size == (224, 224)
```

- [ ] **Step 2: Run the output-contract test and verify RED**

Run:

```bash
pytest -q tests/test_dual_camera_probe_visualization.py \
  -k output_contract
```

Expected: FAIL because `save_dual_camera_sample` does not exist.

- [ ] **Step 3: Implement exact filenames and shared Depth color bounds**

```python
OBSERVATION_NAMES = tuple(
    f"observation_{camera}_{time}.png"
    for camera in CAMERAS for time in ("current", "future")
)
DINO_NAMES = tuple(
    f"dino_{source}_{camera}_{time}_224"
    for source in ("teacher", "planner")
    for camera in CAMERAS for time in ("current", "future")
)
DEPTH_NAMES = tuple(
    f"depth_{source}_{camera}_{time}_224"
    for source in ("moge", "teacher_probe", "planner_probe")
    for camera in CAMERAS for time in ("current", "future")
)
```

In `save_dual_camera_sample`, compute each range only from
`depth_moge_{camera}_{time}_224`, write it to `depth_color_ranges.json`, and
pass the same `(low, high)` to all three Depth render calls for that
camera/time.

- [ ] **Step 4: Write failing saved-probe loading test**

```python
def test_saved_probes_round_trip(tmp_path):
    module = load_module()
    dino = module.DinoPCAProbe(
        torch.zeros(4), torch.eye(4)[:, :3], torch.zeros(3), torch.ones(3)
    )
    torch.save({"state_dict": dino.state_dict()}, tmp_path / "dino_pca_probe.pt")
    depth = module.LinearDepthProbe(feature_dim=4, grid_size=16)
    torch.save(
        {"state_dict": depth.state_dict(), "feature_dim": 4, "grid_size": 16},
        tmp_path / "depth_linear_probe.pt",
    )
    loaded_dino, loaded_depth = module.load_saved_probes(tmp_path, torch.device("cpu"))
    assert torch.equal(loaded_dino.basis, dino.basis)
    assert torch.equal(loaded_depth.projection.weight, depth.projection.weight)
```

- [ ] **Step 5: Run the saved-probe test and verify RED**

Run:

```bash
pytest -q tests/test_dual_camera_probe_visualization.py \
  -k saved_probes_round_trip
```

Expected: FAIL because `load_saved_probes` does not exist.

- [ ] **Step 6: Implement probe loading and the evaluation CLI**

The parser must require the same checkpoint/data/teacher arguments as the
existing evaluation plus:

```python
parser.add_argument("--probe-dir", type=Path, required=True)
parser.add_argument("--train-windows-per-suite", type=int, default=64)
parser.add_argument("--eval-windows-per-suite", type=int, default=16)
parser.add_argument("--planner-batch-size", type=int, default=8)
parser.add_argument("--visualizations-per-suite", type=int, default=2)
```

Load probes with `torch.load(..., map_location="cpu", weights_only=True)`,
reconstruct their modules from saved metadata, and use the same deterministic
`select_disjoint_indices` split. For each batch:

1. Get current/future DINO and Depth teacher outputs.
2. Get current/future planner outputs.
3. Project DINO teacher/planner features separately for main/wrist.
4. Decode Depth teacher/planner/persistence features separately for
   main/wrist using the corresponding MoGe half.
5. Accumulate metrics under `summary[scope][camera][modality][case]`.
6. Save the first two sample directories per suite with the Task 2 contract.

- [ ] **Step 7: Run all local tests and commit production code only**

Run:

```bash
python -m py_compile \
  scripts/qwen3_vl_semantic_planner/visualize_dual_camera_probes.py
pytest -q \
  tests/test_dual_camera_probe_visualization.py \
  tests/test_dino_depth_probe_visualization.py \
  tests/test_depth_probe_visualization.py \
  tests/test_lingbot_planner_evaluation.py
```

Expected: all tests PASS.

Commit only production code:

```bash
git add scripts/qwen3_vl_semantic_planner/visualize_dual_camera_probes.py
git commit -m "feat: visualize aligned dual-camera probes"
```

---

### Task 3: Pod smoke, full camera evaluation, and A6000 synchronization

**Files:**
- Remote output: `outputs/qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_20260714T214246/probe224_dual_camera_20260715/`
- Local output: `artifacts/probe224_dual_camera_20260715/`

**Interfaces:**
- Consumes: Task 2 CLI and remote probes from `probe224_pca_depth_20260715/`.
- Produces: four-suite camera-specific metrics and eight complete sample directories on both Pod and A6000.

- [ ] **Step 1: Sync only the new production script and verify its checksum**

```bash
rsync -av --checksum -e 'ssh -p 30282 -o BatchMode=yes' \
  scripts/qwen3_vl_semantic_planner/visualize_dual_camera_probes.py \
  root@182.242.159.145:/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/
sha256sum scripts/qwen3_vl_semantic_planner/visualize_dual_camera_probes.py
ssh -p 30282 -o BatchMode=yes root@182.242.159.145 \
  'sha256sum /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/visualize_dual_camera_probes.py'
```

Expected: local and remote SHA256 values match.

- [ ] **Step 2: Run a four-suite smoke evaluation**

Use GPU 0 after confirming it is below 500 MiB. Run the new CLI with the same
checkpoint, FastWAM config, four dataset directories, teacher weights, and
environment variables as the previous probe run, plus:

```text
--probe-dir outputs/qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_20260714T214246/probe224_pca_depth_20260715
--output-dir outputs/qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_20260714T214246/probe224_dual_camera_smoke_20260715
--train-windows-per-suite 1
--eval-windows-per-suite 1
--planner-batch-size 1
--visualizations-per-suite 1
```

Expected: exit 0, four sample directories, each containing 24 PNG files,
`instruction.txt`, and `depth_color_ranges.json`; every PNG is 224x224.

- [ ] **Step 3: Run the full reuse-probe evaluation**

Launch with:

```text
--train-windows-per-suite 64
--eval-windows-per-suite 16
--planner-batch-size 8
--visualizations-per-suite 2
--dtype bf16
--device cuda:0
```

Expected: 64 evaluated windows, eight sample directories, finite per-camera
DINO/Depth metrics, and no probe training phase in the log.

- [ ] **Step 4: Verify remote artifact semantics**

Assert:

```text
4 suites
8 sample directories
192 sample PNGs (8 x 24)
8 instruction files
8 depth_color_ranges.json files
all PNG dimensions == 224 x 224
all summary floats are finite
no Traceback, OOM, RuntimeError, or ValueError in the log
```

For every color-range sidecar, verify the three Depth files sharing one
camera/time all reference the same recorded `low` and `high`.

- [ ] **Step 5: Sync results and inspect separate camera outputs**

```bash
mkdir -p artifacts/probe224_dual_camera_20260715
rsync -av --checksum -e 'ssh -p 30282 -o BatchMode=yes' \
  root@182.242.159.145:/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_20260714T214246/probe224_dual_camera_20260715/ \
  artifacts/probe224_dual_camera_20260715/
```

Open, as separate files, main/wrist current/future RGB, DINO teacher/planner,
and Depth MoGe/teacher-probe/planner-probe from the same sample. Confirm camera
identity, spatial correspondence, and the expected loss of fine detail caused
by 16x8-to-224 interpolation.

- [ ] **Step 6: Final verification and code-only push preparation**

Run the Task 2 local test command again. Verify the final production commit
contains only `visualize_dual_camera_probes.py`; keep the design, plan, tests,
and `artifacts/` outside the next GitHub push.

