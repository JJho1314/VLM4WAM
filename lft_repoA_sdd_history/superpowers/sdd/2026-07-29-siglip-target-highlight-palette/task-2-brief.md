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

