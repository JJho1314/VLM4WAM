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

