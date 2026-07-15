# DINO and Depth 224×224 Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a global DINO PCA probe and train a linear relative-depth probe, apply both to teacher and `step_030000` planner tokens, and save separate exact-224×224 visualizations on the A6000.

**Architecture:** Add one focused dual-probe entry point that reuses the tested split, depth target, linear-probe, and checkpoint-loading helpers already present in the repository. The DINO path fits one fixed global PCA coordinate system; the depth path retains the shared per-token linear decoder. Both paths render independent files rather than a query-token contact sheet.

**Tech Stack:** Python 3.10, PyTorch, Pillow, matplotlib, Qwen3-VL, frozen LingBot DINO/Depth teachers, FastWAM LIBERO datasets, pytest, SSH/rsync.

## Global Constraints

- Planner checkpoint is the completed `step_030000` Qwen3-VL-4B LingBot planner.
- Use all four LIBERO suites with 64 deterministic probe-training windows and 16 disjoint probe-evaluation windows per suite.
- Each window contributes current and future-at-offset-8 frames; video augmentation is disabled.
- DINO and Depth token geometry is exactly `[B, 256, 1024]`.
- DINO uses one global three-component PCA basis and fixed training-set 1st/99th-percentile display bounds.
- Depth uses a shared per-token linear `1024 → 1` probe trained with Smooth-L1 plus spatial-gradient loss.
- No convolutional or U-Net decoder and no query-token contact sheet.
- Every saved visualization PNG is exactly 224×224.
- Planner and teacher modules remain frozen.
- Completed artifacts are copied from the Pod to the A6000 workspace.

---

### Task 1: Global DINO PCA probe

**Files:**
- Create: `scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py`
- Create: `tests/test_dino_depth_probe_visualization.py`

**Interfaces:**
- Consumes: teacher or planner feature tensors shaped `[B, 256, 1024]`.
- Produces: `DinoPCAProbe.fit(features: Tensor, seed: int) -> DinoPCAProbe` and `DinoPCAProbe.project_224(features: Tensor) -> Tensor` shaped `[B, 3, 224, 224]` in `[0, 1]`.

- [ ] **Step 1: Write failing PCA tests**

```python
def test_global_dino_pca_is_deterministic_and_outputs_224():
    module = load_module()
    features = torch.randn(6, 256, 12)
    first = module.DinoPCAProbe.fit(features, seed=7)
    second = module.DinoPCAProbe.fit(features, seed=7)
    output = first.project_224(features[:2])
    assert output.shape == (2, 3, 224, 224)
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.basis, second.basis)
    assert torch.isfinite(output).all()
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0


def test_global_dino_pca_does_not_renormalize_each_sample():
    module = load_module()
    training = torch.randn(8, 256, 10)
    probe = module.DinoPCAProbe.fit(training, seed=3)
    base = training[:1]
    shifted = base + 0.5
    assert not torch.allclose(probe.project_224(base), probe.project_224(shifted))
```

- [ ] **Step 2: Run tests and verify the missing interface fails**

Run: `pytest -q tests/test_dino_depth_probe_visualization.py -k dino_pca`

Expected: FAIL because `DinoPCAProbe` is not defined.

- [ ] **Step 3: Implement the minimal fitted PCA module**

```python
class DinoPCAProbe(nn.Module):
    def __init__(self, mean, basis, low, high, output_size=224):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("basis", basis.float())
        self.register_buffer("low", low.float())
        self.register_buffer("high", high.float())
        self.output_size = int(output_size)

    @classmethod
    def fit(cls, features, *, seed=0, output_size=224):
        validate_token_features(features, name="DINO PCA training features")
        flat = features.detach().float().flatten(0, 1).cpu()
        mean = flat.mean(dim=0)
        centered = flat - mean
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            _u, _s, basis = torch.pca_lowrank(centered, q=3, center=False, niter=4)
        projected = centered @ basis[:, :3]
        low = torch.quantile(projected, 0.01, dim=0)
        high = torch.quantile(projected, 0.99, dim=0)
        if not all(torch.isfinite(value).all() for value in (mean, basis, low, high)):
            raise ValueError("DINO PCA probe contains non-finite statistics")
        return cls(mean, basis[:, :3], low, high, output_size=output_size)

    def project_224(self, features):
        validate_token_features(features, feature_dim=self.mean.numel(), name="DINO features")
        projected = (features.float() - self.mean) @ self.basis
        projected = (projected - self.low) / (self.high - self.low).clamp_min(1e-6)
        grid = projected.clamp(0, 1).reshape(-1, 16, 16, 3).permute(0, 3, 1, 2)
        return F.interpolate(
            grid,
            size=(self.output_size, self.output_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)
```

Add `validate_token_features(features, feature_dim=None, name="features")` to
reject non-`[B,256,D]` shapes and non-finite values; when `feature_dim` is not
`None`, also require the exact final dimension. Production extraction calls it
with `feature_dim=1024`, while small unit-test tensors can use another positive
dimension.

- [ ] **Step 4: Run PCA tests**

Run: `pytest -q tests/test_dino_depth_probe_visualization.py -k dino_pca`

Expected: PASS.

- [ ] **Step 5: Commit the PCA probe**

```bash
git add scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py tests/test_dino_depth_probe_visualization.py
git commit -m "feat: add global dino pca probe"
```

### Task 2: Exact-224 depth rendering and separate sample outputs

**Files:**
- Modify: `scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py`
- Modify: `tests/test_dino_depth_probe_visualization.py`

**Interfaces:**
- Consumes: relative log-depth grids `[B, 16, 16]`, 224-pixel targets, RGB observations, instruction text, and projected DINO maps.
- Produces: `decode_depth_224(relative: Tensor, target_depth: Tensor) -> Tensor` and `save_sample_outputs(output_dir: Path, ...) -> list[Path]`.

- [ ] **Step 1: Write failing resize and output-layout tests**

```python
def test_decode_depth_output_is_exactly_224():
    module = load_module()
    relative = torch.randn(2, 16, 16)
    target = torch.rand(2, 256, 256).add_(0.1)
    decoded = module.decode_depth_224(relative, target)
    assert decoded.shape == (2, 224, 224)
    assert torch.isfinite(decoded).all()


def test_sample_outputs_are_separate_and_exactly_224(tmp_path):
    module = load_module()
    paths = module.save_sample_outputs(
        output_dir=tmp_path,
        current_rgb=torch.zeros(224, 448, 3, dtype=torch.uint8),
        future_rgb=torch.zeros(224, 448, 3, dtype=torch.uint8),
        instruction="pick up the bowl",
        dino_maps={name: torch.zeros(3, 224, 224) for name in module.DINO_OUTPUT_NAMES},
        depth_maps={name: torch.ones(224, 224) for name in module.DEPTH_OUTPUT_NAMES},
    )
    assert {path.name for path in paths} == set(module.EXPECTED_SAMPLE_FILES)
    assert (tmp_path / "instruction.txt").read_text() == "pick up the bowl\n"
    for path in paths:
        if path.suffix == ".png":
            assert Image.open(path).size == (224, 224)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest -q tests/test_dino_depth_probe_visualization.py -k 'decode_depth or sample_outputs'`

Expected: FAIL because `decode_depth_224` and `save_sample_outputs` are not defined.

- [ ] **Step 3: Implement 224 decoding and independent files**

Implement `decode_depth_224` by resizing relative log depth to 224×224, resizing the positive target to the same shape, subtracting the predicted spatial mean, and applying the target-log median shift before exponentiation. Implement `save_sample_outputs` with Pillow: take the leftmost 224×224 external-camera view from each 224×448 composite observation, save the instruction as UTF-8 text, convert DINO CHW floats to RGB uint8, and render paired depth files with identical target-derived 2nd/98th-percentile `viridis` bounds.

Use these exact constants:

```python
DINO_OUTPUT_NAMES = (
    "dino_teacher_current_224",
    "dino_planner_current_224",
    "dino_teacher_future_224",
    "dino_planner_future_224",
)
DEPTH_OUTPUT_NAMES = (
    "depth_target_current_224",
    "depth_planner_current_224",
    "depth_target_future_224",
    "depth_planner_future_224",
)
EXPECTED_SAMPLE_FILES = (
    "observation_current.png",
    "observation_future.png",
    "instruction.txt",
    *(f"{name}.png" for name in DINO_OUTPUT_NAMES),
    *(f"{name}.png" for name in DEPTH_OUTPUT_NAMES),
)
```

- [ ] **Step 4: Run output tests**

Run: `pytest -q tests/test_dino_depth_probe_visualization.py -k 'decode_depth or sample_outputs'`

Expected: PASS.

- [ ] **Step 5: Commit rendering support**

```bash
git add scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py tests/test_dino_depth_probe_visualization.py
git commit -m "feat: render separate 224px probe outputs"
```

### Task 3: Dual-teacher training and planner evaluation pipeline

**Files:**
- Modify: `scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py`
- Modify: `tests/test_dino_depth_probe_visualization.py`
- Reuse: `scripts/qwen3_vl_semantic_planner/train_depth_probe_visualization.py`
- Reuse: `scripts/qwen3_vl_semantic_planner/evaluate_lingbot_current_future_planner.py`

**Interfaces:**
- Consumes: the CLI assets listed in the design spec, frozen teacher features, and planner current/future predictions.
- Produces: `dino_pca_probe.pt`, `depth_linear_probe.pt`, histories, overall/per-suite metrics, and per-sample directories.

- [ ] **Step 1: Write failing pipeline-contract tests**

```python
def test_training_cache_contains_both_modalities_and_current_future():
    module = load_module()
    cache = module.ProbeTrainingCache(
        dino=torch.zeros(4, 256, 1024),
        depth=torch.zeros(4, 256, 1024),
        relative_depth=torch.zeros(4, 16, 16),
    )
    cache.validate()


def test_projected_dino_metrics_are_exact_for_equal_maps():
    module = load_module()
    target = torch.rand(2, 3, 224, 224)
    metrics = module.compute_dino_map_metrics(target, target.clone())
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["mean_cosine"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run contract tests and verify failure**

Run: `pytest -q tests/test_dino_depth_probe_visualization.py -k 'training_cache or projected_dino_metrics'`

Expected: FAIL because the cache and metric interfaces are not defined.

- [ ] **Step 3: Implement CLI and frozen extraction**

Add arguments for checkpoint, four repeatable dataset directories, both DINO teacher assets, both depth teacher assets, output directory, split counts, teacher/planner/probe batch sizes, epochs, learning rate, gradient weight, visualization count, device, dtype, and seed.

Build teachers using the existing `DinoVideoTargetEncoder` and `DepthTargetEncoder`. For every training batch, concatenate current/future frames for depth extraction and call DINO `encode_current_and_future` with `future_video_effective_fps[:, 0]`. Store CPU BF16 DINO/Depth features and FP16 16×16 relative-depth targets in `ProbeTrainingCache`.

```python
@dataclass
class ProbeTrainingCache:
    dino: torch.Tensor
    depth: torch.Tensor
    relative_depth: torch.Tensor

    def validate(self) -> None:
        validate_token_features(self.dino, feature_dim=1024, name="cached DINO")
        validate_token_features(self.depth, feature_dim=1024, name="cached Depth")
        if self.dino.shape != self.depth.shape:
            raise ValueError("cached DINO and Depth feature shapes differ")
        if self.relative_depth.shape != (self.dino.shape[0], 16, 16):
            raise ValueError("cached relative depth must be [frames,16,16]")
        if not torch.isfinite(self.relative_depth).all():
            raise ValueError("cached relative depth contains non-finite values")


def extract_teacher_batch(items, dino_teacher, depth_teacher):
    current, future = frames_from_items(items)
    effective_fps = torch.stack(
        [item["future_video_effective_fps"] for item in items]
    )[:, 0]
    current_dino, future_dino = dino_teacher.encode_current_and_future(
        current,
        future,
        effective_fps=effective_fps,
    )
    depth_features, dense_depth = extract_depth_teacher_outputs(
        depth_teacher,
        torch.cat([current, future], dim=0),
    )
    batch = len(items)
    return {
        "dino": torch.cat([current_dino, future_dino], dim=0),
        "depth": depth_features,
        "relative_depth": relative_log_depth(dense_depth, grid_size=16),
        "dense_depth": F.interpolate(
            dense_depth.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False
        )[:, 0],
        "batch_size": batch,
    }
```

- [ ] **Step 4: Fit probes and save artifacts**

Fit `DinoPCAProbe` over all cached teacher DINO features. Reuse `LinearDepthProbe`, `BestProbeStateTracker`, `depth_gradient_loss`, and `relative_log_depth` from `train_depth_probe_visualization.py` to train the depth probe. Save all tensors on CPU with geometry and split metadata.

```python
dino_probe = DinoPCAProbe.fit(cache.dino, seed=args.seed, output_size=224)
depth_probe, history, best = train_depth_probe(
    features=cache.depth,
    targets=cache.relative_depth,
    batch_size=args.probe_batch_size,
    epochs=args.probe_epochs,
    lr=args.probe_lr,
    gradient_loss_weight=args.gradient_loss_weight,
    seed=args.seed,
    device=device,
)
torch.save(
    {
        "state_dict": dino_probe.state_dict(),
        "output_size": 224,
        "token_grid": 16,
        "feature_dim": 1024,
        "seed": args.seed,
    },
    args.output_dir / "dino_pca_probe.pt",
)
torch.save(
    {
        "state_dict": {k: v.detach().cpu() for k, v in depth_probe.state_dict().items()},
        "output_size": 224,
        "token_grid": 16,
        "feature_dim": 1024,
        **best,
    },
    args.output_dir / "depth_linear_probe.pt",
)
```

- [ ] **Step 5: Evaluate teacher and planner tokens**

Load the frozen planner through `_load_runtime`. Use one planner forward to collect all four prediction branches. Project DINO maps through the fixed PCA probe, decode depth maps through the trained linear probe, update overall/per-suite sums, and save only the requested number of separate sample directories. Write `summary.json` and a flat `summary.csv`.

```python
predictions = planner_current_future_predictions(
    items=items,
    wrapper=wrapper,
    processor=processor,
    metadata=metadata,
    device=runtime_device,
    dtype=runtime_dtype,
)
teacher_maps = {
    "dino_teacher_current_224": dino_probe.project_224(teacher_current_dino),
    "dino_teacher_future_224": dino_probe.project_224(teacher_future_dino),
}
planner_maps = {
    "dino_planner_current_224": dino_probe.project_224(predictions["current_dino"]),
    "dino_planner_future_224": dino_probe.project_224(predictions["future_dino"]),
}
depth_maps = {
    "depth_target_current_224": dense_current_depth,
    "depth_planner_current_224": decode_depth_224(
        depth_probe(predictions["current_depth"].float()), dense_current_depth
    ),
    "depth_target_future_224": dense_future_depth,
    "depth_planner_future_224": decode_depth_224(
        depth_probe(predictions["future_depth"].float()), dense_future_depth
    ),
}
```

Implement `compute_dino_map_metrics` as channel-wise cosine averaged over all
pixels plus per-value MSE. Accumulate weighted sums rather than averaging batch
averages. Serialize only finite numeric values.

- [ ] **Step 6: Run all probe tests**

Run: `pytest -q tests/test_depth_probe_visualization.py tests/test_dino_depth_probe_visualization.py tests/test_lingbot_planner_evaluation.py`

Expected: all tests PASS with no failures.

- [ ] **Step 7: Commit the complete local pipeline**

```bash
git add scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py tests/test_dino_depth_probe_visualization.py
git commit -m "feat: train dual 224px planner probes"
```

### Task 4: Pod smoke test, full probe job, and A6000 result sync

**Files:**
- Remote create: `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py`
- Remote output: `/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_20260714T214246/probe224_pca_depth_20260715`
- Local output: `artifacts/probe224_pca_depth_20260715`

**Interfaces:**
- Consumes: the tested local script and all existing Pod assets.
- Produces: verified full probe artifacts on the Pod and A6000.

- [ ] **Step 1: Run local syntax and focused regression checks**

Run:

```bash
python -m py_compile scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py
pytest -q tests/test_depth_probe_visualization.py tests/test_dino_depth_probe_visualization.py tests/test_lingbot_planner_evaluation.py
```

Expected: compilation exits 0 and all tests PASS.

- [ ] **Step 2: Sync only required scripts and verify checksums**

Run:

```bash
remote=root@182.242.159.145
port=30282
remote_dir=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner
rsync -av --checksum -e "ssh -p $port -o BatchMode=yes" \
  scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py \
  scripts/qwen3_vl_semantic_planner/train_depth_probe_visualization.py \
  scripts/qwen3_vl_semantic_planner/evaluate_lingbot_current_future_planner.py \
  "$remote:$remote_dir/"
sha256sum \
  scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py \
  scripts/qwen3_vl_semantic_planner/train_depth_probe_visualization.py \
  scripts/qwen3_vl_semantic_planner/evaluate_lingbot_current_future_planner.py
ssh -p "$port" -o BatchMode=yes "$remote" \
  "sha256sum $remote_dir/train_dino_depth_probe_visualization.py $remote_dir/train_depth_probe_visualization.py $remote_dir/evaluate_lingbot_current_future_planner.py"
```

Expected: all three local/remote SHA256 pairs match.

- [ ] **Step 3: Run one-window smoke**

Launch on one idle H100 with all four suites, `--train-windows-per-suite 1`, `--eval-windows-per-suite 1`, `--probe-epochs 2`, batch sizes 1, and one visualization per suite. Require exit code 0, finite summaries, 4 sample directories, and 44 sample files.

Run the following command after `nvidia-smi` confirms GPU 0 is idle:

```bash
ssh -tt -p 30282 -o BatchMode=yes root@182.242.159.145 '
repo=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713
run=qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_20260714T214246
cd "$repo" || exit 2
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false XFORMERS_DISABLED=1
export PYTHONPATH="$repo/third_party/FastWAM/src${PYTHONPATH:+:$PYTHONPATH}"
export LINGBOT_SRC_ROOT=/root/nas/junjie/code/lingbot-vla-v2
export UTILS3D_MOGE_PATH=/root/nas/junjie/py_deps/utils3d_moge
export FASTWAM_FRAME_CACHE_DIR=/root/nas/junjie/data/LIBERO-fastwam/frame_cache_224
/opt/conda/envs/vlm4wam/bin/python scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py \
  --checkpoint-dir "outputs/$run/step_030000" \
  --fastwam-data-config third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_spatial_no_noops_lerobot \
  --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_object_no_noops_lerobot \
  --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_goal_no_noops_lerobot \
  --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_10_no_noops_lerobot \
  --fastwam-text-embedding-cache-dir /root/nas/junjie/data/libero_qwen \
  --fastwam-pretrained-norm-stats /root/nas/junjie/data/LIBERO-fastwam_meta/dataset_stats.json \
  --dino-teacher-ckpt /root/nas/junjie/weights/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth \
  --dino-teacher-config /root/nas/junjie/weights/lingbot-vla-v2-6b/dino_video/config.yaml \
  --depth-moge-path /root/nas/junjie/weights/moge-2-vitb-normal/model.pt \
  --depth-morgbd-path /root/nas/junjie/weights/lingbot-vla-v2-6b/depth/model.pt \
  --output-dir "outputs/$run/probe224_pca_depth_smoke_20260715" \
  --train-windows-per-suite 1 --eval-windows-per-suite 1 \
  --teacher-batch-size 1 --planner-batch-size 1 --probe-batch-size 1 \
  --probe-epochs 2 --probe-lr 0.003 --gradient-loss-weight 0.2 \
  --visualizations-per-suite 1 --dtype bf16 --device cuda:0
'
```

- [ ] **Step 4: Launch the full probe job**

Use one H100 with:

```text
--train-windows-per-suite 64
--eval-windows-per-suite 16
--teacher-batch-size 8
--planner-batch-size 8
--probe-batch-size 64
--probe-epochs 100
--probe-lr 0.003
--gradient-loss-weight 0.2
--visualizations-per-suite 2
--dtype bf16
```

Capture PID and log path, then monitor cache extraction, probe epochs, and 64 evaluation windows to completion.

Run:

```bash
ssh -p 30282 -o BatchMode=yes root@182.242.159.145 '
set -euo pipefail
repo=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713
run=qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_20260714T214246
out=$repo/outputs/$run/probe224_pca_depth_20260715
log=$repo/logs/probe224_pca_depth_20260715.log
cd "$repo"
nohup env \
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false XFORMERS_DISABLED=1 \
  PYTHONPATH="$repo/third_party/FastWAM/src${PYTHONPATH:+:$PYTHONPATH}" \
  LINGBOT_SRC_ROOT=/root/nas/junjie/code/lingbot-vla-v2 \
  UTILS3D_MOGE_PATH=/root/nas/junjie/py_deps/utils3d_moge \
  FASTWAM_FRAME_CACHE_DIR=/root/nas/junjie/data/LIBERO-fastwam/frame_cache_224 \
  /opt/conda/envs/vlm4wam/bin/python scripts/qwen3_vl_semantic_planner/train_dino_depth_probe_visualization.py \
  --checkpoint-dir "outputs/$run/step_030000" \
  --fastwam-data-config third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_spatial_no_noops_lerobot \
  --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_object_no_noops_lerobot \
  --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_goal_no_noops_lerobot \
  --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_10_no_noops_lerobot \
  --fastwam-text-embedding-cache-dir /root/nas/junjie/data/libero_qwen \
  --fastwam-pretrained-norm-stats /root/nas/junjie/data/LIBERO-fastwam_meta/dataset_stats.json \
  --dino-teacher-ckpt /root/nas/junjie/weights/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth \
  --dino-teacher-config /root/nas/junjie/weights/lingbot-vla-v2-6b/dino_video/config.yaml \
  --depth-moge-path /root/nas/junjie/weights/moge-2-vitb-normal/model.pt \
  --depth-morgbd-path /root/nas/junjie/weights/lingbot-vla-v2-6b/depth/model.pt \
  --output-dir "$out" \
  --train-windows-per-suite 64 --eval-windows-per-suite 16 \
  --teacher-batch-size 8 --planner-batch-size 8 --probe-batch-size 64 \
  --probe-epochs 100 --probe-lr 0.003 --gradient-loss-weight 0.2 \
  --visualizations-per-suite 2 --dtype bf16 --device cuda:0 \
  > "$log" 2>&1 < /dev/null &
pid=$!
printf "PID=%s\nLOG=%s\nOUTPUT=%s\n" "$pid" "$log" "$out"
'
```

- [ ] **Step 5: Verify remote artifacts**

Assert `summary.json` reports four suites, 256 training windows, and 64 evaluation windows. Assert two probe checkpoints exist, metrics are finite, eight visualized sample directories exist, each directory has 11 files, and every PNG is exactly 224×224. Scan the log for traceback, OOM, and non-finite values; confirm the process exited and GPU memory is released.

- [ ] **Step 6: Copy results to A6000 and verify parity**

Use checksum-based `rsync` to copy the full output directory to `artifacts/probe224_pca_depth_20260715/`. Compare remote/local PNG counts, file sizes, and probe SHA256 hashes. Open representative DINO and Depth current/future outputs for visual inspection.

Run:

```bash
mkdir -p artifacts/probe224_pca_depth_20260715
rsync -av --checksum -e 'ssh -p 30282 -o BatchMode=yes' \
  root@182.242.159.145:/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_20260714T214246/probe224_pca_depth_20260715/ \
  artifacts/probe224_pca_depth_20260715/
find artifacts/probe224_pca_depth_20260715 -type f -name '*.png' | wc -l
sha256sum artifacts/probe224_pca_depth_20260715/dino_pca_probe.pt \
  artifacts/probe224_pca_depth_20260715/depth_linear_probe.pt
```

Expected: 81 PNG files (eight sample directories × ten PNGs plus one global
training-curve PNG), two non-empty
probe hashes, and no checksum differences reported by a second `rsync` dry run.
