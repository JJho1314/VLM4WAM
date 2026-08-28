# Task 2 report — Fixed-Phase Palette Comparison Generator

## TDD evidence

- Primitive RED: focused pytest failed with `ModuleNotFoundError` for the absent comparison generator.
- Primitive GREEN: focused pytest passed `6 passed` after phase, palette, and overlay primitives were added.
- Integration RED: focused pytest failed with absent `generate_comparison` import.
- Integration GREEN: focused pytest passed `7 passed` with an injected fake highlighter.

## Delivered behavior

- Added the fixed frame-128 target switch, exact channel-permutation palettes, relevance overlay, 4x4 Pillow grid, CLI, collision refusal, and same-directory temporary-file publication.
- The integration test creates all main/wrist inputs for frames 112 and 160, checks phrase order and PNG/JSON metadata, preserves SHA-256 hashes for every RGB/probe source, and confirms outputs are the only added paths.

## Verification

`/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest tests/test_siglip2_target_highlight.py tests/test_siglip2_pca_probe.py tests/test_export_libero_episode_siglip2_da3.py -q` — `57 passed`.

`ruff check ...`, `py_compile ...`, and `git diff --check` all exited successfully.

## Self-review

Checked source existence precedes highlighter construction; both final files reject collisions; the highlighter runs once over row-ordered RGB images and each relevance map is reused across A/B/C.

## Runtime fix

- Ola exposed that exported RGB reference images are 512x512 while `siglip_probe.png` is 256x256.
- RED: changing the integration fixture RGB inputs to 512x512 reproduced the pre-model `ValueError`; GREEN: the focused suite passed after RGB-only loading was changed to deterministic Pillow `Image.Resampling.LANCZOS` resizing to 256x256.
- Probe loading remains strict: every `siglip_probe.png` must already be exactly 256x256.
- Re-ran the full required command after the fix: `57 passed`; ruff, py_compile, and `git diff --check` passed.
