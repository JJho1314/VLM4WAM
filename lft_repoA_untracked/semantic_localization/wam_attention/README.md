# FastWAM RGB-only vs SG-WAM cross-attention (local A6000)

- `final_viz.py` — builds SG FastWAM (loads step_015000.pt) + baseline (base video DiT + SG text_proj),
  captures video→text cross-attention (blocks 8/12/16/20, mean over valid Qwen tokens) on LIBERO frames,
  saves maps to wam_maps.npz.
- `render_sharp.py` — re-renders the 2-row poster from wam_maps.npz with percentile+gamma contrast.
- Output: ../figs/wam_rgbonly_vs_sg.png
- SG checkpoint (5.2G, pulled from Ola) + Cosmos-2B weights live OUTSIDE the repo at
  /data/LFT-W02_data/junjie/fastwam_sg_ckpt/ and /data/LFT-W02_data/junjie/weights/Cosmos-Predict2.5-2B.
- Runs in the cosmos venv: /data/LFT-W02_data/junjie/cosmos-predict2.5/.venv, using the PATCHED
  cosmos at VLA_WM/VLM4WAM/cosmos-predict2.5 (has semantic_plan_context).

## Batch generate + select (64 samples)
- `generate_many.py rgb` / `generate_many.py sg` — run ONE model per process (isolation avoids the
  two-2B-in-one-process crash), 64 LIBERO samples (≤2 frames/task), saves wam_part_{rgb,sg}.npz.
- `select_wam.py` — scores each sample by concentration-gain (SG top-10% mass − RGB, minus edge
  penalty), ranks, renders the TOP-8 clearest cases → ../figs/wam_sg_best.png.
- SG concentration (0.19–0.31) is consistently > RGB-only (0.14–0.17): SG-WAM attention is focused,
  RGB-only is diffuse.

## Initial-frame scenes (more scenes, per request)
- `extract_init_frames.py` — reads LIBERO lerobot episodes, extracts the INITIAL (frame-0)
  observation (main|wrist -> [224,448] composite) for many episodes per task -> libsamples_init.npz
  (240 scenes, 40 tasks x 6 episodes = distinct initial object layouts).
- `generate_many.py {rgb,sg}` now reads libsamples_init.npz; incremental checkpoint save every 40
  (the shared box's external OOM pressure kills the RGB run ~step 180, so RGB has 160 / SG has 240;
  they share sample order, so the first 160 are paired).
- `select_wam.py` aligns to the shorter, ranks by concentration-gain -> ../figs/wam_sg_best_init.png (top-10).
- Result: on initial frames too, SG concentration (0.20-0.25) > RGB-only (0.14-0.17).
