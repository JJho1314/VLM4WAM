# LIBERO Episode SigLIP2 and DA3 Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export independently viewable RGB, globally comparable SigLIP2 PCA, and globally normalized DA3 depth PNGs at stride 16 over LIBERO episode 288.

**Architecture:** Add one focused CLI exporter beside the existing SigLIP2/DA3 visualization utilities. Keep episode indexing, output naming, and visualization normalization as pure functions with CPU tests; keep model imports lazy so the tests do not need model weights. The CLI decodes each requested frame once, batches teacher inference, then writes one directory per camera and sampled frame plus a manifest.

**Tech Stack:** Python 3.11, PyTorch, PyAV, Pillow, Matplotlib, Transformers SigLIP2, Depth Anything 3, pytest.

## Global Constraints

- Dataset suite is `libero_10_no_noops_lerobot`.
- Episode is `episode_000288`, with 224 frames at 20 FPS.
- Sampling stride is 16 and must include both frame 0 and frame 223.
- Cameras are `observation.images.image` (`main`) and `observation.images.wrist_image` (`wrist`).
- Each camera/frame directory contains exactly `rgb.png`, `siglip_pca.png`, and `da3_depth.png`.
- SigLIP2 PCA uses one basis and one display range jointly across all sampled frames and cameras.
- DA3 disparity uses one robust display range jointly across all sampled frames and cameras.
- Planner predictions and composite panel figures are excluded.
- Successful output contains 15 sampled frames, 2 cameras, and 90 PNG files.

---

### Task 1: Episode sampling and artifact layout

**Files:**
- Create: `tests/test_export_libero_episode_siglip2_da3.py`
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py`

**Interfaces:**
- Produces: `sample_frame_indices(num_frames: int, stride: int) -> list[int]`
- Produces: `artifact_paths(output_dir: Path, camera: str, frame_index: int) -> dict[str, Path]`
- Consumes: only Python standard-library types.

- [ ] **Step 1: Write failing tests for inclusive sampling and output paths**

```python
from pathlib import Path

import pytest

from qwen3_vl_semantic_planner.dinov3_da3_2b.export_libero_episode_siglip2_da3 import (
    artifact_paths,
    sample_frame_indices,
)


def test_sample_frame_indices_covers_episode_end() -> None:
    assert sample_frame_indices(224, 16) == [
        0, 16, 32, 48, 64, 80, 96, 112,
        128, 144, 160, 176, 192, 208, 223,
    ]


@pytest.mark.parametrize(("num_frames", "stride"), [(0, 16), (224, 0), (224, -1)])
def test_sample_frame_indices_rejects_invalid_inputs(
    num_frames: int,
    stride: int,
) -> None:
    with pytest.raises(ValueError):
        sample_frame_indices(num_frames, stride)


def test_artifact_paths_separates_camera_frame_and_modalities(tmp_path: Path) -> None:
    assert artifact_paths(tmp_path, "wrist", 32) == {
        "rgb": tmp_path / "wrist/frame_000032/rgb.png",
        "siglip_pca": tmp_path / "wrist/frame_000032/siglip_pca.png",
        "da3_depth": tmp_path / "wrist/frame_000032/da3_depth.png",
    }
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
/opt/miniconda3/bin/python -m pytest tests/test_export_libero_episode_siglip2_da3.py -q
```

Expected: collection fails with `ModuleNotFoundError` for `export_libero_episode_siglip2_da3`.

- [ ] **Step 3: Implement the two pure helpers**

```python
from pathlib import Path


def sample_frame_indices(num_frames: int, stride: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    indices = list(range(0, num_frames, stride))
    final_index = num_frames - 1
    if indices[-1] != final_index:
        indices.append(final_index)
    return indices


def artifact_paths(
    output_dir: Path,
    camera: str,
    frame_index: int,
) -> dict[str, Path]:
    frame_dir = output_dir / camera / f"frame_{frame_index:06d}"
    return {
        "rgb": frame_dir / "rgb.png",
        "siglip_pca": frame_dir / "siglip_pca.png",
        "da3_depth": frame_dir / "da3_depth.png",
    }
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
/opt/miniconda3/bin/python -m pytest tests/test_export_libero_episode_siglip2_da3.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Commit Task 1**

```bash
git add tests/test_export_libero_episode_siglip2_da3.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py
git commit -m "feat(viz): define full-episode sampling layout"
```

---

### Task 2: Globally comparable SigLIP2 and DA3 visualizations

**Files:**
- Modify: `tests/test_export_libero_episode_siglip2_da3.py`
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py`

**Interfaces:**
- Produces: `siglip_pca_images(features: torch.Tensor, grid_size: int, output_size: int) -> np.ndarray`
- Produces: `da3_depth_images(depth: torch.Tensor) -> np.ndarray`
- Consumes: SigLIP features shaped `[frames, grid_size * grid_size, feature_dim]`.
- Consumes: positive DA3 depth shaped `[frames, height, width]`.
- Returns: RGB `uint8` arrays shaped `[frames, output_size, output_size, 3]` for SigLIP and `[frames, height, width, 3]` for DA3.

- [ ] **Step 1: Add failing behavior tests**

```python
import numpy as np
import torch

from qwen3_vl_semantic_planner.dinov3_da3_2b.export_libero_episode_siglip2_da3 import (
    da3_depth_images,
    siglip_pca_images,
)


def test_siglip_pca_images_use_one_transform_for_all_frames() -> None:
    base = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    features = torch.stack((base, base), dim=0)
    images = siglip_pca_images(features, grid_size=2, output_size=8)
    assert images.shape == (2, 8, 8, 3)
    assert images.dtype == np.uint8
    np.testing.assert_array_equal(images[0], images[1])


def test_da3_depth_images_share_one_episode_color_scale() -> None:
    depth = torch.stack(
        (torch.full((4, 4), 1.0), torch.full((4, 4), 2.0)),
        dim=0,
    )
    images = da3_depth_images(depth)
    assert images.shape == (2, 4, 4, 3)
    assert images.dtype == np.uint8
    assert not np.array_equal(images[0], images[1])


def test_siglip_pca_images_reject_non_square_token_count() -> None:
    with pytest.raises(ValueError, match="grid"):
        siglip_pca_images(torch.zeros(2, 5, 4), grid_size=2, output_size=8)


def test_da3_depth_images_reject_non_positive_depth() -> None:
    with pytest.raises(ValueError, match="positive"):
        da3_depth_images(torch.zeros(2, 4, 4))
```

- [ ] **Step 2: Run the tests and verify missing symbols**

Run:

```bash
/opt/miniconda3/bin/python -m pytest tests/test_export_libero_episode_siglip2_da3.py -q
```

Expected: collection fails because `siglip_pca_images` and `da3_depth_images` do not exist.

- [ ] **Step 3: Implement shared PCA and depth color normalization**

```python
import numpy as np
import torch
import torch.nn.functional as F


def _robust_unit_interval(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1, values.shape[-1])
    low = torch.quantile(flat, 0.02, dim=0)
    high = torch.quantile(flat, 0.98, dim=0)
    return ((values - low) / (high - low + 1e-6)).clamp(0, 1)


def siglip_pca_images(
    features: torch.Tensor,
    *,
    grid_size: int,
    output_size: int,
) -> np.ndarray:
    if features.ndim != 3 or features.shape[1] != grid_size * grid_size:
        raise ValueError("SigLIP token count must match grid_size squared")
    x = features.detach().float().cpu()
    flat = x.reshape(-1, x.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
    projected = (centered @ vectors[:3].T).reshape(
        x.shape[0], grid_size, grid_size, 3
    )
    projected = _robust_unit_interval(projected)
    resized = F.interpolate(
        projected.permute(0, 3, 1, 2),
        size=(output_size, output_size),
        mode="nearest",
    ).permute(0, 2, 3, 1)
    return (resized * 255.0).round().to(torch.uint8).numpy()


def da3_depth_images(depth: torch.Tensor) -> np.ndarray:
    if depth.ndim != 3 or not torch.all(depth > 0):
        raise ValueError("DA3 depth must be positive [frames,height,width]")
    disparity = depth.detach().float().cpu().reciprocal()
    low = torch.quantile(disparity, 0.02)
    high = torch.quantile(disparity, 0.98)
    normalized = ((disparity - low) / (high - low + 1e-6)).clamp(0, 1)
    from matplotlib import colormaps
    rgb = colormaps["turbo"](normalized.numpy())[..., :3]
    return np.rint(rgb * 255.0).astype(np.uint8)
```

- [ ] **Step 4: Run focused and related tests**

Run:

```bash
/opt/miniconda3/bin/python -m pytest \
  tests/test_export_libero_episode_siglip2_da3.py \
  tests/test_ge_act_dual_camera_planner.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add tests/test_export_libero_episode_siglip2_da3.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py
git commit -m "feat(viz): add global SigLIP and DA3 color mapping"
```

---

### Task 3: Episode decoding, teacher inference, and independent PNG export

**Files:**
- Modify: `tests/test_export_libero_episode_siglip2_da3.py`
- Modify: `qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py`

**Interfaces:**
- Produces: `load_episode_record(data_root: Path, suite: str, episode_index: int) -> dict`
- Produces: `decode_episode_frames(video_path: Path, frame_indices: list[int]) -> np.ndarray`
- Produces: `write_export(output_dir: Path, frames: np.ndarray, siglip_rgb: np.ndarray, depth_rgb: np.ndarray, camera_names: tuple[str, ...], frame_indices: list[int]) -> list[dict]`
- Produces CLI flags: `--data-root`, `--suite`, `--episode-index`, `--stride`, `--siglip2-model-dir`, `--da3-ckpt-dir`, `--da3-code-root`, `--output-dir`, `--batch-size`, `--device`.

- [ ] **Step 1: Add failing metadata and file-writing tests**

```python
import json
from PIL import Image

from qwen3_vl_semantic_planner.dinov3_da3_2b.export_libero_episode_siglip2_da3 import (
    load_episode_record,
    write_export,
)


def test_load_episode_record_returns_exact_episode(tmp_path: Path) -> None:
    meta = tmp_path / "suite/meta"
    meta.mkdir(parents=True)
    (meta / "episodes.jsonl").write_text(
        '{"episode_index": 7, "tasks": ["task seven"], "length": 9}\n'
        '{"episode_index": 8, "tasks": ["task eight"], "length": 10}\n',
        encoding="utf-8",
    )
    assert load_episode_record(tmp_path, "suite", 8) == {
        "episode_index": 8,
        "tasks": ["task eight"],
        "length": 10,
    }


def test_write_export_creates_three_independent_pngs_per_camera_frame(
    tmp_path: Path,
) -> None:
    frames = np.zeros((2, 2, 4, 4, 3), dtype=np.uint8)
    siglip = np.full((4, 8, 8, 3), 64, dtype=np.uint8)
    depth = np.full((4, 4, 4, 3), 128, dtype=np.uint8)
    records = write_export(
        tmp_path,
        frames=frames,
        siglip_rgb=siglip,
        depth_rgb=depth,
        camera_names=("main", "wrist"),
        frame_indices=[0, 8],
    )
    assert len(records) == 4
    assert len(list(tmp_path.rglob("*.png"))) == 12
    assert Image.open(tmp_path / "main/frame_000000/rgb.png").size == (4, 4)
    assert Image.open(tmp_path / "main/frame_000000/siglip_pca.png").size == (8, 8)
```

- [ ] **Step 2: Run tests and verify missing symbols**

Run:

```bash
/opt/miniconda3/bin/python -m pytest tests/test_export_libero_episode_siglip2_da3.py -q
```

Expected: collection fails because `load_episode_record` and `write_export` do not exist.

- [ ] **Step 3: Implement metadata lookup, exact frame decoding, image writing, and manifest records**

Implement these contracts:

```python
def load_episode_record(data_root: Path, suite: str, episode_index: int) -> dict:
    episodes_path = data_root / suite / "meta/episodes.jsonl"
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["episode_index"]) == episode_index:
            return record
    raise KeyError(f"episode {episode_index} not found in {episodes_path}")


def decode_episode_frames(
    video_path: Path,
    frame_indices: list[int],
) -> np.ndarray:
    import av
    requested = set(frame_indices)
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in requested:
                decoded[index] = frame.to_ndarray(format="rgb24")
            if len(decoded) == len(requested):
                break
    missing = [index for index in frame_indices if index not in decoded]
    if missing:
        raise RuntimeError(f"{video_path} is missing requested frames {missing}")
    return np.stack([decoded[index] for index in frame_indices])
```

`write_export` must flatten camera-major feature arrays in the same order used
for model inference, create every frame directory, save each array with
`PIL.Image.fromarray(...).save(...)`, and return records containing camera,
frame index, timestamp in seconds, and relative paths for all three files.

- [ ] **Step 4: Implement lazy teacher loading and batched inference in `main()`**

The CLI must:

1. Read the exact episode record and `meta/info.json`.
2. Build both video paths from `chunks_size` and episode index.
3. Decode `[0, 16, ..., 223]` for both cameras.
4. Flatten frames in camera-major order to `[2 * 15, 3, H, W]`.
5. Instantiate `Siglip2TargetEncoder` with `input_size=256`, `grid_size=16`,
   and encode batches through its public `encode_future_keyframes` method.
6. Instantiate the DA3 target helper and full DA3 model, then reuse
   `DepthAnything3TargetEncoder._prep`, `full.model.backbone`, and
   `full.model._process_depth_head` in batches.
7. Fit global SigLIP PCA and global DA3 disparity ranges only after all
   features/depth maps are available.
8. Write 90 PNGs and a `manifest.json`.
9. Assert the exported PNG count is
   `len(frame_indices) * len(camera_names) * 3` before printing a JSON
   `{"status": "done", ...}` record.

- [ ] **Step 5: Run the full local test selection**

Run:

```bash
/opt/miniconda3/bin/python -m pytest \
  tests/test_export_libero_episode_siglip2_da3.py \
  tests/test_ge_act_dual_camera_planner.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Verify CLI syntax without loading models**

Run:

```bash
/opt/miniconda3/bin/python \
  qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py \
  --help
```

Expected: exit 0 and all nine CLI flags appear.

- [ ] **Step 7: Commit Task 3**

```bash
git add tests/test_export_libero_episode_siglip2_da3.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py
git commit -m "feat(viz): export a sampled LIBERO episode"
```

---

### Task 4: HPC3 execution and artifact verification

**Files:**
- Runtime output: `/data/user/jhe724/junjie/outputs/libero_episode_000288_siglip2_da3_stride16/`
- Local synchronized output: `outputs/libero_episode_000288_siglip2_da3_stride16/`

**Interfaces:**
- Consumes the Task 3 CLI.
- Produces 90 PNG files and one `manifest.json`.

- [ ] **Step 1: Check HPC3 debug partition and required assets**

Run:

```bash
ssh hpc3 "sinfo -p debug -o '%P %a %l %D %G'; \
test -d /data/user/jhe724/junjie/datasets/LIBERO-fastwam; \
test -d /data/user/jhe724/junjie/weights/siglip2-large-patch16-256; \
test -d /data/user/jhe724/junjie/vlm4wam_joint_assets/DA3-LARGE-1.1; \
test -d /data/user/jhe724/junjie/vlm4wam_joint_assets/Depth-Anything-3"
```

Expected: debug partition is up and all tests exit 0.

- [ ] **Step 2: Sync only the exporter to the existing HPC3 worktree**

Run:

```bash
rsync -av \
  qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py \
  hpc3:/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af/qwen3_vl_semantic_planner/dinov3_da3_2b/
```

Expected: one Python file transferred.

- [ ] **Step 3: Submit a one-GPU debug job**

The Slurm command must request `debug`, one GPU, at most 12 CPUs, 128 GB RAM,
and 30 minutes. Invoke:

```bash
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python \
  qwen3_vl_semantic_planner/dinov3_da3_2b/export_libero_episode_siglip2_da3.py \
  --data-root /data/user/jhe724/junjie/datasets/LIBERO-fastwam \
  --suite libero_10_no_noops_lerobot \
  --episode-index 288 \
  --stride 16 \
  --siglip2-model-dir /data/user/jhe724/junjie/weights/siglip2-large-patch16-256 \
  --da3-ckpt-dir /data/user/jhe724/junjie/vlm4wam_joint_assets/DA3-LARGE-1.1 \
  --da3-code-root /data/user/jhe724/junjie/vlm4wam_joint_assets/Depth-Anything-3 \
  --output-dir /data/user/jhe724/junjie/outputs/libero_episode_000288_siglip2_da3_stride16 \
  --batch-size 8 \
  --device cuda
```

- [ ] **Step 4: Monitor by output condition**

Poll Slurm and the output directory at intervals no longer than 60 seconds.
Stop when the job reaches a terminal state. Read the complete log and require
`"status": "done"` plus Slurm `COMPLETED` with exit code `0:0`.

- [ ] **Step 5: Synchronize lightweight artifacts to the workspace**

Run:

```bash
mkdir -p outputs/libero_episode_000288_siglip2_da3_stride16
rsync -av \
  hpc3:/data/user/jhe724/junjie/outputs/libero_episode_000288_siglip2_da3_stride16/ \
  outputs/libero_episode_000288_siglip2_da3_stride16/
```

- [ ] **Step 6: Verify artifact completeness**

Run:

```bash
/opt/miniconda3/bin/python -c '
import json
from pathlib import Path
p = Path("outputs/libero_episode_000288_siglip2_da3_stride16")
m = json.loads((p / "manifest.json").read_text())
assert m["frame_indices"] == [0,16,32,48,64,80,96,112,128,144,160,176,192,208,223]
assert len(list(p.rglob("*.png"))) == 90
assert len(m["records"]) == 30
for record in m["records"]:
    for relative in record["files"].values():
        assert (p / relative).is_file(), relative
print("EPISODE_EXPORT_OK", len(m["records"]), len(list(p.rglob("*.png"))))
'
```

Expected: `EPISODE_EXPORT_OK 30 90`.

- [ ] **Step 7: Visually inspect representative outputs**

Open at least these six independent files:

- `main/frame_000000/rgb.png`
- `main/frame_000112/siglip_pca.png`
- `main/frame_000223/da3_depth.png`
- `wrist/frame_000000/rgb.png`
- `wrist/frame_000112/siglip_pca.png`
- `wrist/frame_000223/da3_depth.png`

Confirm each is a single modality image, not a composite, and that later
frames show plausible trajectory changes.
