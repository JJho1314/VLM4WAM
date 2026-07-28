# SigLIP Target-Highlight Palette Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one directly viewable PNG that compares three PCA channel palettes while highlighting only the currently manipulated cup in both the main and wrist cameras.

**Architecture:** Add a focused SigLIP target-relevance module that derives a 16-by-16 Grad-CAM map from the full frozen SigLIP image-text score, then add a standalone comparison generator that combines that map with the existing learned `siglip_probe.png`. Run the diagnostic on Ola, using a fixed frame-128 target switch, and sync only the new comparison PNG and provenance JSON back beside the existing episode export.

**Tech Stack:** Python 3.10, PyTorch, Transformers `SiglipModel`, Pillow, NumPy, pytest, Ola H100 runtime

## Global Constraints

- Source episode is exactly `libero_10_no_noops_lerobot/episode_000288`.
- Instruction is exactly `put the white mug on the left plate and put the yellow and white mug on the right plate`.
- `frame_index < 128` uses target phrase `the white textured mug`.
- `frame_index >= 128` uses target phrase `the yellow and white mug`.
- Both the main and wrist cameras are required.
- Both cameras use the same active-target phase for a given frame, but compute separate relevance maps.
- Relevance must come from the complete frozen `siglip2-large-patch16-256` image-text score and Grad-CAM over penultimate vision tokens.
- Do not compute raw penultimate-patch/text cosine similarity.
- The learned PCA probe checkpoint and all existing 120 exported PNG files must remain unchanged.
- Palette A is `[PC1, PC2, PC3]`, B is `[PC2, PC3, PC1]`, and C is `[PC3, PC1, PC2]`.
- Initial comparison frames are exactly 112 and 160.
- The output contains four rows: frame-112 main, frame-112 wrist, frame-160 main, frame-160 wrist.
- The output contains four columns: RGB, palette A highlight, palette B highlight, palette C highlight.
- The generated PNG and JSON sidecar live under `target_highlight_comparison/` beside the episode export.
- No SigLIP or probe parameter may be updated.

---

### Task 1: Frozen SigLIP Pairwise Grad-CAM

**Files:**
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_target_highlight.py`
- Create: `tests/test_siglip2_target_highlight.py`

**Interfaces:**
- Consumes: a local `siglip2-large-patch16-256` checkpoint, RGB images as `PIL.Image.Image`, and one target phrase per image.
- Produces:
  - `normalize_relevance(cam: torch.Tensor, low_q: float = 0.05, high_q: float = 0.95) -> torch.Tensor`
  - `token_gradcam(tokens: torch.Tensor, gradients: torch.Tensor, *, grid_size: int = 16, output_size: int = 256) -> torch.Tensor`
  - `SiglipPairGradCAM(model_dir: Path, device: torch.device)`
  - `SiglipPairGradCAM(images: Sequence[Image.Image], phrases: Sequence[str]) -> np.ndarray`, returning `[B, 256, 256]` float32 relevance in `[0, 1]`.

- [ ] **Step 1: Write normalization and spatial Grad-CAM tests**

Add literal, hand-derived tests:

```python
def test_normalize_relevance_uses_fixed_quantiles_and_handles_zero_range():
    cam = torch.arange(100, dtype=torch.float32).reshape(1, 10, 10)
    normalized = normalize_relevance(cam, low_q=0.05, high_q=0.95)
    assert normalized.shape == (1, 10, 10)
    assert normalized.min().item() == 0.0
    assert normalized.max().item() == 1.0
    torch.testing.assert_close(
        normalize_relevance(torch.ones(1, 4, 4)),
        torch.zeros(1, 4, 4),
    )


def test_token_gradcam_keeps_positive_activation_gradient_product():
    tokens = torch.zeros(1, 4, 2)
    gradients = torch.zeros_like(tokens)
    tokens[0, 3] = torch.tensor([2.0, 1.0])
    gradients[0, 3] = torch.tensor([1.0, 2.0])
    cam = token_gradcam(
        tokens,
        gradients,
        grid_size=2,
        output_size=2,
    )
    assert cam.shape == (1, 2, 2)
    assert cam[0, 1, 1] == 1.0
    assert torch.count_nonzero(cam) == 1
```

- [ ] **Step 2: Run the focused tests and witness RED**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_target_highlight.py -q
```

Expected: collection or import failure because
`siglip2_target_highlight.py` does not exist.

- [ ] **Step 3: Implement the pure Grad-CAM functions**

Implement:

```python
def normalize_relevance(
    cam: torch.Tensor,
    low_q: float = 0.05,
    high_q: float = 0.95,
) -> torch.Tensor:
    flat = cam.flatten(1)
    low = torch.quantile(flat, low_q, dim=1).view(-1, 1, 1)
    high = torch.quantile(flat, high_q, dim=1).view(-1, 1, 1)
    span = high - low
    normalized = torch.where(
        span > 0,
        (cam - low) / span.clamp_min(torch.finfo(cam.dtype).eps),
        torch.zeros_like(cam),
    )
    return normalized.clamp(0, 1)


def token_gradcam(
    tokens: torch.Tensor,
    gradients: torch.Tensor,
    *,
    grid_size: int = 16,
    output_size: int = 256,
) -> torch.Tensor:
    expected = grid_size * grid_size
    if tokens.shape != gradients.shape or tokens.ndim != 3:
        raise ValueError("tokens and gradients must share [B,N,D]")
    if tokens.shape[1] != expected:
        raise ValueError(f"expected {expected} tokens")
    cam = (tokens.float() * gradients.float()).sum(dim=-1).relu()
    cam = cam.reshape(tokens.shape[0], 1, grid_size, grid_size)
    cam = torch.nn.functional.interpolate(
        cam,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return normalize_relevance(cam)
```

- [ ] **Step 4: Run the pure-function tests and witness GREEN**

Run the command from Step 2.

Expected: the two tests pass.

- [ ] **Step 5: Write a fake-model autograd contract test**

Create a small real `torch.nn.Module` fake whose penultimate token tensor
drives a pairwise `[B, B]` similarity matrix. The test must prove:

- image `i` uses diagonal score `[i, i]`;
- returned maps have `[B, 256, 256]`;
- model parameters have `requires_grad=False`;
- calling the highlighter does not populate parameter gradients;
- unequal image/phrase counts raise `ValueError`;
- the highlighter requests vision hidden states and selects
  `vision_model_output.hidden_states[-2]`.

Use dependency injection in the constructor:

```python
class SiglipPairGradCAM:
    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
        *,
        model: torch.nn.Module | None = None,
        processor: Any | None = None,
    ) -> None:
        ...
```

The fake processor returns literal `pixel_values`, `input_ids`, and
`attention_mask`; the fake model returns a `SimpleNamespace` with
`logits_per_image` and `vision_model_output.hidden_states`.

- [ ] **Step 6: Run the fake-model test and witness RED**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_target_highlight.py -q
```

Expected: failure because `SiglipPairGradCAM` is absent or incomplete.

- [ ] **Step 7: Implement the frozen pairwise highlighter**

Load `AutoModel` and `AutoProcessor` from the same local checkpoint with
`local_files_only=True`. Validate:

- model class exposes `vision_model` and `text_model`;
- native vision size is 256;
- patch size is 16;
- hidden size is 1024.

Freeze all parameters with `requires_grad_(False)` and set eval mode.

On each call:

```python
inputs = processor(
    text=list(phrases),
    images=list(images),
    padding="max_length",
    return_tensors="pt",
)
pixel_values = inputs.pop("pixel_values").to(device)
pixel_values.requires_grad_(True)
outputs = model(
    pixel_values=pixel_values,
    **{name: value.to(device) for name, value in inputs.items()},
    output_hidden_states=True,
    return_dict=True,
)
tokens = outputs.vision_model_output.hidden_states[-2]
tokens.retain_grad()
score = outputs.logits_per_image.diagonal().sum()
score.backward()
if tokens.grad is None:
    raise RuntimeError("SigLIP penultimate tokens did not receive gradients")
maps = token_gradcam(tokens.detach(), tokens.grad.detach())
model.zero_grad(set_to_none=True)
pixel_values.grad = None
return maps.cpu().numpy().astype(np.float32)
```

Do not wrap this path in `torch.no_grad()` or `torch.inference_mode()`.

- [ ] **Step 8: Run Task 1 tests and static checks**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_target_highlight.py -q
ruff check \
  qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_target_highlight.py \
  tests/test_siglip2_target_highlight.py
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m py_compile \
  qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_target_highlight.py
git diff --check
```

Expected: all pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add \
  qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_target_highlight.py \
  tests/test_siglip2_target_highlight.py
git commit -m "feat(viz): add SigLIP target Grad-CAM"
```

---

### Task 2: Fixed-Phase Palette Comparison Generator

**Files:**
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/generate_siglip_target_highlight_comparison.py`
- Modify: `tests/test_siglip2_target_highlight.py`

**Interfaces:**
- Consumes:
  - Task 1 `SiglipPairGradCAM`;
  - existing episode-export directory containing
    `{main,wrist}/frame_{000112,000160}/{rgb,siglip_probe}.png`;
  - a local SigLIP2 model directory.
- Produces:
  - `active_target(frame_index: int) -> str`
  - `permute_palette(feature_rgb: np.ndarray, order: tuple[int, int, int]) -> np.ndarray`
  - `combine_target_highlight(feature_rgb: np.ndarray, relevance: np.ndarray) -> np.ndarray`
  - `generate_comparison(export_root: Path, model_dir: Path, output_dir: Path, device: torch.device) -> tuple[Path, Path]`
  - `siglip_target_highlight_palettes.png`
  - `siglip_target_highlight_palettes.json`

- [ ] **Step 1: Write phase, palette, and overlay tests**

Add:

```python
def test_active_target_switches_at_frame_128():
    assert active_target(112) == "the white textured mug"
    assert active_target(127) == "the white textured mug"
    assert active_target(128) == "the yellow and white mug"
    assert active_target(160) == "the yellow and white mug"


def test_palette_candidates_are_exact_channel_permutations():
    pixel = np.array([[[10, 20, 30]]], dtype=np.uint8)
    np.testing.assert_array_equal(
        permute_palette(pixel, (0, 1, 2)),
        np.array([[[10, 20, 30]]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        permute_palette(pixel, (1, 2, 0)),
        np.array([[[20, 30, 10]]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        permute_palette(pixel, (2, 0, 1)),
        np.array([[[30, 10, 20]]], dtype=np.uint8),
    )


def test_combined_highlight_preserves_shape_and_emphasizes_target():
    feature = np.full((8, 8, 3), [40, 120, 200], dtype=np.uint8)
    relevance = np.zeros((8, 8), dtype=np.float32)
    relevance[2:6, 2:6] = 1.0
    combined = combine_target_highlight(feature, relevance)
    assert combined.shape == (8, 8, 3)
    assert combined.dtype == np.uint8
    assert combined[3, 3].sum() > combined[0, 0].sum()
    assert combined[2, 2, 0] > combined[0, 0, 0]
```

- [ ] **Step 2: Run the focused tests and witness RED**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_target_highlight.py -q
```

Expected: imports fail because the comparison generator is absent.

- [ ] **Step 3: Implement phase, palette, and overlay primitives**

Define exact constants:

```python
PHASE_BOUNDARY = 128
TARGET_BEFORE = "the white textured mug"
TARGET_AFTER = "the yellow and white mug"
PALETTES = {
    "A_current": (0, 1, 2),
    "B_warm_balanced": (1, 2, 0),
    "C_cool_balanced": (2, 0, 1),
}
CAMERAS = ("main", "wrist")
FRAMES = (112, 160)
```

For `combine_target_highlight`:

1. Convert feature RGB to float `[0, 1]`.
2. Compute grayscale luminance with literal coefficients
   `[0.2126, 0.7152, 0.0722]`.
3. Form the background as
   `0.75 * (0.40 * color + 0.60 * grayscale)`.
4. Blend background toward the original color by relevance.
5. Add warm-yellow `[1.0, 0.72, 0.0]` with alpha
   `0.28 * relevance`.
6. Define the binary target as `relevance >= 0.65`.
7. Compute a one-pixel contour as max-pool minus min-pool and color it
   amber `[1.0, 0.55, 0.0]`.
8. Round, clip, and return uint8.

- [ ] **Step 4: Run primitive tests and witness GREEN**

Run the Task 2 Step 2 command.

Expected: all current tests pass.

- [ ] **Step 5: Write the generator integration test**

Build a temporary export with four literal 256-by-256 RGB/probe PNG pairs:

- main frame 112;
- wrist frame 112;
- main frame 160;
- wrist frame 160.

Inject a fake relevance provider:

```python
class FakeHighlighter:
    def __call__(self, images, phrases):
        maps = np.zeros((len(images), 256, 256), dtype=np.float32)
        maps[:, 64:192, 64:192] = 1.0
        return maps
```

Call `generate_comparison(..., highlighter=FakeHighlighter())` and assert:

- the fake receives phrases in exact row order:
  first target twice, second target twice;
- the output PNG is readable and has four rows by four columns;
- the JSON contains model path, frame boundary, frames, cameras, phrases,
  quantiles, palette mappings, and panel order;
- no source RGB or `siglip_probe.png` hash changes;
- no extra files appear outside the requested output directory.

- [ ] **Step 6: Run the integration test and witness RED**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_target_highlight.py -q
```

Expected: failure because `generate_comparison` is absent or incomplete.

- [ ] **Step 7: Implement the comparison generator and CLI**

Use Pillow and the installed Chinese font:

`/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc`

Render a white-background 4-by-4 panel grid with:

- 256-by-256 image content;
- a 44-pixel column-label band;
- a 36-pixel row-label band;
- 12-pixel gutters;
- column labels `RGB`, `A · 当前`, `B · 暖色`, `C · 冷色`;
- row labels containing frame, camera, and active phrase.

Keep panel generation separate from relevance computation. Load the four RGB
images in row order, compute all four Grad-CAM maps in one call, and reuse
each row's map across A/B/C.

Expose:

```text
--export-root
--siglip2-model-dir
--output-dir
--device
```

Require every source path before loading the model. Refuse to overwrite either
final output file. Write the PNG and JSON through same-directory temporary
files followed by `Path.replace`.

- [ ] **Step 8: Run Task 2 and related regression tests**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_target_highlight.py \
  tests/test_siglip2_pca_probe.py \
  tests/test_export_libero_episode_siglip2_da3.py -q
ruff check \
  qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_target_highlight.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/generate_siglip_target_highlight_comparison.py \
  tests/test_siglip2_target_highlight.py
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m py_compile \
  qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_target_highlight.py \
  qwen3_vl_semantic_planner/dinov3_da3_2b/generate_siglip_target_highlight_comparison.py
git diff --check
```

Expected: all pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add \
  qwen3_vl_semantic_planner/dinov3_da3_2b/generate_siglip_target_highlight_comparison.py \
  tests/test_siglip2_target_highlight.py
git commit -m "feat(viz): compare phase-aware SigLIP palettes"
```

---

### Task 3: Ola Generation, Sync, and Visual Validation

**Files:**
- Runtime output:
  `/data/users/junjie/outputs/libero_episode_000288_siglip2_da3_stride16_probe/target_highlight_comparison/siglip_target_highlight_palettes.png`
- Runtime sidecar:
  `/data/users/junjie/outputs/libero_episode_000288_siglip2_da3_stride16_probe/target_highlight_comparison/siglip_target_highlight_palettes.json`
- Local output directory:
  `/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/libero_episode_000288_siglip2_da3_stride16_probe/target_highlight_comparison`

**Interfaces:**
- Consumes: exact Task 1/2 source at reviewed HEAD, Ola H100, existing remote
  episode export, and local SigLIP2 model weights.
- Produces: one verified PNG and one JSON sidecar, with no modification to
  existing episode files.

- [ ] **Step 1: Verify local source and tests**

Run the Task 2 Step 8 test and static-check commands at the reviewed HEAD.

- [ ] **Step 2: Preflight Ola without mutation**

Run:

```bash
ssh olabots '
  hostname
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
  test -x /data/users/junjie/envs/vlm4wam/bin/python
  test -d /data/users/junjie/vlm4wam_2b/weights/siglip2-large-patch16-256
  test -d /data/users/junjie/outputs/libero_episode_000288_siglip2_da3_stride16_probe
'
```

Select a GPU with less than 500 MiB in use. Stop if no GPU is available.

- [ ] **Step 3: Snapshot the original remote export**

Before creating the new subdirectory, hash exactly the existing 120 PNGs and
the manifest:

```bash
ssh olabots '
  cd /data/users/junjie/outputs/libero_episode_000288_siglip2_da3_stride16_probe
  find main wrist -type f -name "*.png" -print0 |
    sort -z |
    xargs -0 sha256sum
  sha256sum manifest.json
'
```

Save the output under `/data/users/junjie/logs/` with a task-specific name.

- [ ] **Step 4: Sync only the two diagnostic source files**

Use `rsync` to deploy:

```text
siglip2_target_highlight.py
generate_siglip_target_highlight_comparison.py
```

to:

`/data/users/junjie/code/VLM4WAM_dual_camera_k4/qwen3_vl_semantic_planner/dinov3_da3_2b/`

Verify local and remote SHA256 pairs match exactly.

- [ ] **Step 5: Run the generator on Ola**

Use GPU 0 when it remains free:

```bash
ssh olabots '
  cd /data/users/junjie/code/VLM4WAM_dual_camera_k4
  CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/data/users/junjie/code/VLM4WAM_dual_camera_k4 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  /data/users/junjie/envs/vlm4wam/bin/python -u \
    qwen3_vl_semantic_planner/dinov3_da3_2b/generate_siglip_target_highlight_comparison.py \
    --export-root /data/users/junjie/outputs/libero_episode_000288_siglip2_da3_stride16_probe \
    --siglip2-model-dir /data/users/junjie/vlm4wam_2b/weights/siglip2-large-patch16-256 \
    --output-dir /data/users/junjie/outputs/libero_episode_000288_siglip2_da3_stride16_probe/target_highlight_comparison \
    --device cuda
'
```

Capture stdout/stderr at:

`/data/users/junjie/logs/siglip-target-highlight-palettes.log`

- [ ] **Step 6: Verify remote artifacts and immutability**

Require:

- process exit code 0;
- one non-empty readable RGB PNG;
- one valid JSON sidecar;
- JSON phase boundary equals 128;
- JSON camera list is exactly `["main", "wrist"]`;
- JSON frames are exactly `[112, 160]`;
- JSON palette mappings match the Global Constraints;
- the post-run hash snapshot of the original 120 PNGs and manifest matches
  Step 3 byte-for-byte.

- [ ] **Step 7: Sync the comparison locally**

Use:

```bash
rsync -av \
  olabots:/data/users/junjie/outputs/libero_episode_000288_siglip2_da3_stride16_probe/target_highlight_comparison/ \
  /data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/libero_episode_000288_siglip2_da3_stride16_probe/target_highlight_comparison/
```

Verify remote and local SHA256 hashes match.

- [ ] **Step 8: Inspect the full comparison image**

Open the local PNG with the image viewer. Confirm:

- frame 112 main and wrist emphasize the white textured mug rather than the
  yellow-white mug;
- frame 160 main and wrist emphasize the yellow-white mug rather than the
  placed first mug;
- every row has RGB plus all three palette candidates;
- palette candidates differ only by channel order;
- non-target feature structure remains visible;
- labels are readable without zooming.

If either target is not localized, preserve the diagnostic output and report
the failed row. Do not silently substitute raw patch/text cosine or an
unapproved external model.

- [ ] **Step 9: Final verification**

Run:

```bash
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest \
  tests/test_siglip2_target_highlight.py \
  tests/test_siglip2_pca_probe.py \
  tests/test_export_libero_episode_siglip2_da3.py -q
git status --short
```

Report:

- local comparison PNG path;
- Ola comparison PNG path;
- selected phase boundary and target phrases;
- source SHA;
- artifact SHA;
- visual-localization result for all four rows;
- confirmation that both cameras are included and the original 120 PNGs are
  unchanged.
