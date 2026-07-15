# MoRGBD Reference-Style Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate truthful 4B MoRGBD planner results in the exact 3-by-6 visual structure of the supplied 2B reference image.

**Architecture:** Keep inference, probes, panels, and metrics unchanged. Make the renderer styling an explicit, tested contract, rerun the four-suite evaluator on the Pod, then copy verified artifacts back locally.

**Tech Stack:** Python 3, PyTorch, Pillow, Matplotlib, pytest, rsync, SSH

## Global Constraints

- Keep the planner, MiniDPT probe, DINO PCA probe, MoRGBD features, and MoGe teacher unchanged.
- Keep truthful labels `MoGe-full` and `depth_absrel`.
- Use a white `20 × 10.5` inch canvas, 110 DPI, tight crop, square panels, 9-point panel titles, and an 11-point two-line figure title.
- Truncate the displayed instruction to 80 characters.
- Generate separate `sample_XX_main.png` and `sample_XX_wrist.png` grids.
- Keep standalone panels at 224 by 224 pixels and `rgb_guidance: false` in the summary.
- Do not push process documents, tests, or generated artifacts.

---

### Task 1: Lock the reference renderer contract

**Files:**
- Modify: `tests/test_morgbd_minidpt_visualization.py`
- Modify: `scripts/qwen3_vl_semantic_planner/visualize_morgbd_minidpt_probe.py:147-225`

**Interfaces:**
- Consumes: `save_reference_style_sample(...)` and the existing named 224-by-224 panels.
- Produces: `_reference_suptitle(camera: str, instruction: str, metrics: dict[str, float]) -> str` and opaque main/wrist PNG grids.

- [ ] **Step 1: Write the failing title-format test**

```python
def test_reference_suptitle_matches_supplied_style() -> None:
    title = _reference_suptitle(
        camera="main",
        instruction="x" * 90,
        metrics={
            "dino_current_mse": 0.0049,
            "dino_future_mse": 0.0073,
            "depth_current_abs_rel": 0.355,
            "depth_future_abs_rel": 0.387,
        },
    )
    assert title == (
        f"[main cam] {'x' * 80}\n"
        "dino_mse cur=0.0049 fut=0.0073  |  "
        "depth_absrel cur=0.355 fut=0.387"
    )
```

- [ ] **Step 2: Run RED**

Run `pytest -q tests/test_morgbd_minidpt_visualization.py::test_reference_suptitle_matches_supplied_style`.

Expected: collection fails because `_reference_suptitle` does not exist.

- [ ] **Step 3: Implement the minimal title formatter and opaque figure settings**

```python
def _reference_suptitle(*, camera, instruction, metrics):
    return (
        f"[{camera} cam] {instruction[:80]}\n"
        f"dino_mse cur={metrics['dino_current_mse']:.4f} "
        f"fut={metrics['dino_future_mse']:.4f}  |  "
        f"depth_absrel cur={metrics['depth_current_abs_rel']:.3f} "
        f"fut={metrics['depth_future_abs_rel']:.3f}"
    )
```

Create the subplots with `facecolor="white"`, set each axis face color to white, call `_reference_suptitle`, and save with `facecolor="white", transparent=False` while preserving `figsize=(20, 10.5)`, `dpi=110`, and `bbox_inches="tight"`.

- [ ] **Step 4: Write and run the image-contract test**

Generate the two grids with white dummy panels and assert:

```python
with Image.open(tmp_path / "sample_02_main.png") as image:
    assert image.size == (2181, 1137)
    assert image.mode == "RGBA"
    assert image.getpixel((0, image.height - 1)) == (255, 255, 255, 255)
```

Run `pytest -q tests/test_morgbd_minidpt_visualization.py`.

Expected: all tests pass.

- [ ] **Step 5: Run legacy visualization tests and commit locally**

Run:

```bash
pytest -q tests/test_morgbd_minidpt_visualization.py tests/test_depth_probe_visualization.py tests/test_dino_depth_probe_visualization.py tests/test_dual_camera_probe_visualization.py tests/test_lingbot_planner_evaluation.py
git add scripts/qwen3_vl_semantic_planner/visualize_morgbd_minidpt_probe.py tests/test_morgbd_minidpt_visualization.py
git commit -m "fix: match MoRGBD grids to reference style"
```

Expected: all tests pass. The test change remains local and is excluded when preparing the final code-only commit.

---

### Task 2: Regenerate and verify main/wrist figures

**Files:**
- Sync: `scripts/qwen3_vl_semantic_planner/visualize_morgbd_minidpt_probe.py`
- Create remotely: `outputs/morgbd_minidpt_reference_style_20260715/`
- Copy locally: `artifacts/morgbd_minidpt_reference_style_20260715/`

**Interfaces:**
- Consumes: planner `step_030000`, `minidpt_depth_probe.pt`, `dino_pca_probe.pt`, four LIBERO suites, and frozen DINO/MoRGBD/MoGe teachers.
- Produces: 16 reference grids, 192 standalone panels, `summary.json`, and `summary.csv`.

- [ ] **Step 1: Sync the renderer**

```bash
rsync -av --checksum -e "ssh -p 30282" scripts/qwen3_vl_semantic_planner/visualize_morgbd_minidpt_probe.py root@182.242.159.145:/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/scripts/qwen3_vl_semantic_planner/
```

- [ ] **Step 2: Run the four-suite evaluator on Pod GPU 0**

Use the exact validated inputs below and a new output directory:

```bash
ssh -p 30282 root@182.242.159.145 'cd /root/nas/junjie/code/VLM4WAM_k1_zero2_20260713 && CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false XFORMERS_DISABLED=1 PYTHONPATH=/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/third_party/FastWAM/src LINGBOT_SRC_ROOT=/root/nas/junjie/code/lingbot-vla-v2 UTILS3D_MOGE_PATH=/root/nas/junjie/py_deps/utils3d_moge FASTWAM_FRAME_CACHE_DIR=/root/nas/junjie/data/LIBERO-fastwam/frame_cache_224 /opt/conda/envs/vlm4wam/bin/python scripts/qwen3_vl_semantic_planner/visualize_morgbd_minidpt_probe.py --checkpoint-dir outputs/qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_20260714T214246/step_030000 --minidpt-probe outputs/morgbd_minidpt_depth_v2_20260715/minidpt_depth_probe.pt --dino-probe outputs/qwen3vl4b_lingbot_sharedhead_q64_dynamicfps_zero2_k1_b16a1_s30000_cached_20260714T214246/probe224_pca_depth_20260715/dino_pca_probe.pt --fastwam-data-config third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_spatial_no_noops_lerobot --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_object_no_noops_lerobot --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_goal_no_noops_lerobot --fastwam-dataset-dir /root/nas/junjie/data/LIBERO-fastwam/libero_10_no_noops_lerobot --fastwam-text-embedding-cache-dir /root/nas/junjie/data/libero_qwen --fastwam-pretrained-norm-stats /root/nas/junjie/data/LIBERO-fastwam_meta/dataset_stats.json --dino-teacher-ckpt /root/nas/junjie/weights/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth --dino-teacher-config /root/nas/junjie/weights/lingbot-vla-v2-6b/dino_video/config.yaml --depth-moge-path /root/nas/junjie/weights/moge-2-vitb-normal/model.pt --depth-morgbd-path /root/nas/junjie/weights/lingbot-vla-v2-6b/depth/model.pt --output-dir outputs/morgbd_minidpt_reference_style_20260715 --train-windows-per-suite 256 --eval-windows-per-suite 16 --planner-batch-size 8 --visualizations-per-suite 2 --dtype bf16 --device cuda:0 --seed 20260715'
```

Expected: `phase=complete`, 64 evaluated windows, four suites, and `rgb_guidance=false`.

- [ ] **Step 3: Verify and copy artifacts**

Assert 16 grids at 2181 by 1137 with opaque white corners, 192 standalone panels at 224 by 224, and a false `summary["protocol"]["rgb_guidance"]`. Then copy:

```bash
rsync -av --checksum -e "ssh -p 30282" root@182.242.159.145:/root/nas/junjie/code/VLM4WAM_k1_zero2_20260713/outputs/morgbd_minidpt_reference_style_20260715/ artifacts/morgbd_minidpt_reference_style_20260715/
```

Expected: remote/local `summary.json` SHA-256 values match.

- [ ] **Step 4: Inspect and finish**

Open one main and one wrist grid from `artifacts/morgbd_minidpt_reference_style_20260715/samples/libero_object/`. Verify the reference layout, white blanks, correct title formatting, camera separation, and unchanged predictions. Then run:

```bash
python -m py_compile scripts/qwen3_vl_semantic_planner/visualize_morgbd_minidpt_probe.py
pytest -q tests/test_morgbd_minidpt_visualization.py tests/test_depth_probe_visualization.py tests/test_dino_depth_probe_visualization.py tests/test_dual_camera_probe_visualization.py tests/test_lingbot_planner_evaluation.py
git diff --check
```

Expected: compilation and all tests pass with no whitespace errors. Rewrite the local task commits so the final pushed commit contains only `scripts/qwen3_vl_semantic_planner/visualize_morgbd_minidpt_probe.py`, then push `lingbot-zero2-q64-k1` normally without force.
