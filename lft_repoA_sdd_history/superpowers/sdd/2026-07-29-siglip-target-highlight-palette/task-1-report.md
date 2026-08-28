# Task 1 Report — Frozen SigLIP Pairwise Grad-CAM

## Implementation

- Added `siglip2_target_highlight.py` with per-map quantile normalization and spatial
  activation-times-gradient Grad-CAM from the native 16x16 visual token grid.
- Added `SiglipPairGradCAM`, which locally loads the full SigLIP model and processor,
  validates the `256 / 16 / 1024` vision contract, freezes/evals all parameters, and
  backpropagates only the sum of `logits_per_image` diagonal entries.
- Uses `vision_model_output.hidden_states[-2]`, clears model and input gradients after
  producing CPU `float32` maps in `[0, 1]`.

## Tests and TDD evidence

1. RED (pure helpers): focused pytest collection failed with
   `ModuleNotFoundError: ...siglip2_target_highlight` before production code existed.
2. GREEN (pure helpers): focused pytest passed `2 passed in 1.32s` after the minimal
   helper implementation.
3. RED (pairwise highlighter): focused pytest collection failed with
   `ImportError: cannot import name 'SiglipPairGradCAM'` before the highlighter existed.
4. GREEN (full focused suite): `3 passed in 1.49s`.

The fake-model autograd test verifies diagonal image/text pairing through distinct top-left
and bottom-right relevance regions, requests hidden states, relies on the penultimate state
instead of a disconnected final-state decoy, freezes parameters, leaves parameter gradients
empty, checks output shape/dtype, and rejects mismatched image/phrase counts.

## Verification

- `ruff check qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_target_highlight.py tests/test_siglip2_target_highlight.py` — passed.
- `/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m py_compile qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_target_highlight.py` — passed.
- `git diff --check` — passed.

## Files

- Added `qwen3_vl_semantic_planner/dinov3_da3_2b/siglip2_target_highlight.py`
- Added `tests/test_siglip2_target_highlight.py`

## Self-review

- The implementation never uses raw patch/text cosine; relevance is differentiated from
  the full model's pairwise score matrix diagonal.
- The Grad-CAM execution path intentionally has no `no_grad` or `inference_mode` context.
- Scope is isolated to the requested new module and test; existing camera/export/training
  code is unchanged.

## Runtime Fix Round 1 (Transformers 4.57)

- Runtime source inspection found that `SiglipVisionTransformer.forward` discards encoder hidden
  states, so `vision_model_output.hidden_states` is `None` despite `output_hidden_states=True`.
- Replaced that unavailable-output access with a forward hook registered immediately before the
  full SigLIP call on `model.vision_model.encoder.layers[-2]`; it unwraps tuple output and is
  removed in `finally` before diagonal pairwise-score backpropagation.
- The fake now mirrors the real encoder-layer structure and returns `hidden_states=None`. RED
  reproduced the production `TypeError: 'NoneType' object is not subscriptable`; GREEN then
  passed `10 passed in 1.75s`, including zero/multiple capture rejection and hook cleanup after
  both successful and failing full-model calls. Ruff, py_compile, and diff checks also passed.
