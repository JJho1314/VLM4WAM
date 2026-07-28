# 2B planner feature probes and visualization

These utilities decode frozen-teacher or planner-predicted feature tokens into
human-readable 224x224 panels. Probe checkpoints and generated images are runtime
artifacts and are not committed to git.

## WSA four-layer path

The dual-camera K4 planner aligns each future keyframe to SigLIP2 and to four DA3
backbone layers (`11,15,19,23`). Its depth predictions therefore have layout
`[B,V,K*256,4,2048]`, while the online teacher targets use
`[B,V,4,K*256,2048]`.

- `wsa_depth_probe.py` defines `WSAMultiLayerDPTProbe`. It applies non-affine,
  per-token LayerNorm, maps the four 16x16 DA3 feature maps to DPT scales, and
  decodes one 224x224 log-depth map.
- `train_feature_probes.py --which da3_wsa` trains that decoder only on frozen,
  real DA3 features and full DA3 depth-head targets. It writes
  `da3_depth_wsa_probe.pt` and records the layer order and normalization contract.
- `visualize_qwen3vl2b_siglip2_da3_dual_camera_k4.py` reads the planner metadata,
  selects the WSA probe for `da3_align_strategy=wsa_multilayer`, and preserves the
  camera and keyframe axes until rendering.

Example on OLA:

```bash
export FASTWAM_FRAME_CACHE_DIR=/data/users/junjie/data/frame_cache/libero
export DA3_CKPT_DIR=/data/users/junjie/vlm4wam_2b/weights/DA3-LARGE-1.1
export DA3_CODE_ROOT=/data/users/junjie/vlm4wam_2b/code/Depth-Anything-3
export DA3_TEACHER_LAYERS=11,15,19,23

python qwen3_vl_semantic_planner/dinov3_da3_2b/train_feature_probes.py \
  --which da3_wsa --steps 5000 --batch-size 16 \
  --out-dir /data/users/junjie/probes_2b

python qwen3_vl_semantic_planner/dinov3_da3_2b/\
visualize_qwen3vl2b_siglip2_da3_dual_camera_k4.py \
  --checkpoint-dir /path/to/step_030000 \
  --siglip2-model-dir /data/users/junjie/vlm4wam_2b/weights/siglip2-large-patch16-256 \
  --da3-ckpt-dir "$DA3_CKPT_DIR" --da3-code-root "$DA3_CODE_ROOT" \
  --output-dir outputs/viz_dual_camera_k4_wsa_step030000
```

For three samples, the visualizer writes 24 PNGs (three samples, two cameras,
four future keyframes) plus `manifest.json`. Every PNG is a 3x6 grid containing
current/future RGB, joint target/prediction SigLIP2 PCA, four-layer depth probe
decodes, token error maps, and full-DA3 reference depth.

## Legacy last-layer path

Legacy metadata (`da3_align_strategy=last_layer`, or no strategy field) still loads
`da3_depth_v2_probe.pt` through `MiniDPTProbe`. Existing probe choices and filenames
remain unchanged:

- `dino_rgb_probe.pt`
- `dino_upsample_probe.pt`
- `da3_depth_probe.pt`
- `da3_depth_v2_probe.pt`

The legacy split renderer is
`visualize_qwen3vl2b_siglip2_da3_split.py`. Its single-layer softness relative to
the full DA3 depth head is a visualization bottleneck and does not describe the
quality of a four-layer WSA checkpoint.

## Runtime artifact locations

The default OLA probe directory is `/data/users/junjie/probes_2b`. The keeper WSA
probe is `da3_depth_wsa_probe.pt`; generated visualizations should remain under an
ignored `outputs/` directory.

## SigLIP2 PCA upsampling probe

Input: penultimate 16x16x1024 SigLIP2-large tokens. Target: the same frozen
model at 512 input, fixed global PCA, 32x32 -> 256x256. The probe is
feature-only and does not accept RGB.

On HPC3, submit the reproducible formal recipe with:

```bash
sbatch qwen3_vl_semantic_planner/dinov3_da3_2b/\
sbatch_train_siglip2_pca_probe_hpc3.sh
```

The launcher uses the `acd_u` partition with one GPU and 12 CPUs. Set
`RUN_KIND=smoke` to force 2 steps, 1 PCA batch, and 1 validation batch; the
formal default is 5,000 steps, batch size 8, 25 PCA batches, and 50 validation
batches. Its environment overrides are `RUN_KIND`, `STEPS`, `BATCH_SIZE`,
`OUTPUT_DIR`, `FRAME_CACHE_DIR`, `SIGLIP2_MODEL_DIR`, `PYTHON`, and
`REPO_ROOT`.

Keeper: `siglip2_pca_upsample_probe.pt`. Rejected run:
`siglip2_pca_upsample_probe_rejected.pt` and exit code 2.
To add an accepted probe to episode export, pass it explicitly:

```bash
python qwen3_vl_semantic_planner/dinov3_da3_2b/\
export_libero_episode_siglip2_da3.py \
  --siglip-pca-probe /path/to/siglip2_pca_upsample_probe.pt
```
