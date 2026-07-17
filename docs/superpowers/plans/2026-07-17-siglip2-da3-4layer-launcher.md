# SigLIP2 + DA3 Four-Layer Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated SigLIP2 + DA3 launcher that aligns DA3 layers 11, 15, 19, and 23 without modifying the existing last-layer experiment record.

**Architecture:** Create one derived outer launcher that keeps every baseline setting unchanged except the output directory and the three DA3 alignment variables. It delegates to the existing `train_lingbot_dino_4b.sh`, which already translates the four-layer variables into trainer arguments.

**Tech Stack:** Bash, existing Qwen3-VL planner trainer, static shell assertions.

## Global Constraints

- Keep `qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3.sh` byte-for-byte unchanged; its SHA-256 before implementation is `9d9e05b46ab7868684b02f465e0829285a40916580e5ed13d9ef114c7fbbc9f0`.
- Name the new launcher `launch_pod_2b_siglip2da3_4layer.sh`; its filename must not contain `wsa`.
- Use `qwen3vl2b_siglip2_da3_4layer_libero_cur_k1` as the default output directory; its name must not contain `wsa`.
- Pass `DA3_ALIGN_STRATEGY=wsa_multilayer`, `DA3_TEACHER_LAYERS=11,15,19,23`, and `DA3_LAYER_WEIGHTS=1.0,1.2,1.4,1.6` explicitly.
- Keep the SigLIP2 teacher as `siglip2-large-patch16-256` and preserve all other baseline settings.
- Do not change model, trainer, visualization, or existing launcher code.

---

### Task 1: Add the Four-Layer Launcher

**Files:**
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3_4layer.sh`
- Reference only: `qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3.sh`
- Reference only: `qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh`
- Test: one-off static shell contract; no permanent test file

**Interfaces:**
- Consumes: environment variables accepted by `train_lingbot_dino_4b.sh`, specifically `DA3_ALIGN_STRATEGY`, `DA3_TEACHER_LAYERS`, and `DA3_LAYER_WEIGHTS`.
- Produces: an executable outer launcher whose default output is isolated from the last-layer baseline.

- [ ] **Step 1: Run the static contract and verify it fails because the new launcher is absent**

Run:

```bash
launcher=qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3_4layer.sh
test -f "$launcher" \
  && test -x "$launcher" \
  && bash -n "$launcher" \
  && rg -q 'OUTPUT_DIR=.*qwen3vl2b_siglip2_da3_4layer_libero_cur_k1' "$launcher" \
  && rg -q 'DA3_ALIGN_STRATEGY=wsa_multilayer' "$launcher" \
  && rg -q 'DA3_TEACHER_LAYERS=11,15,19,23' "$launcher" \
  && rg -q 'DA3_LAYER_WEIGHTS=1.0,1.2,1.4,1.6' "$launcher"
```

Expected: exit status `1` at `test -f` because the new launcher does not exist.

- [ ] **Step 2: Add the derived launcher with the complete configuration**

Use `apply_patch` to create the file with exactly this content:

```bash
#!/usr/bin/env bash
# Derived four-layer variant of launch_pod_2b_siglip2da3.sh for the 2B SigLIP2+DA3
# LIBERO planner on pod 30332 (8xH100). The sibling launcher remains the verbatim
# last-layer experiment record; this file changes only the DA3 alignment and output name.
# Like the baseline record, this launcher is pod-specific and uses /data/users/junjie paths.
#
# Four-layer DA3 alignment uses backbone layers 11,15,19,23 with progressively larger
# loss weights. The internal trainer API calls this strategy "wsa_multilayer"; launcher
# and output names use the experiment-facing "4layer" label.
set -euo pipefail
J=/data/users/junjie
ROOT=$J
REPO_ROOT=$J/code/VLM4WAM_k1_zero2_bidir
DATA_ROOT=/data/shared/datasets/libero_fastwam
W2B=$J/vlm4wam_2b/weights
PY=$J/envs/vlm4wam/bin/python
RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=${NUM_GPUS:-8}; BATCH_SIZE=${BATCH_SIZE:-32}; GRAD_ACCUM=${GRAD_ACCUM:-1}
if [[ "$RUN_KIND" == "smoke" ]]; then MAX_STEPS=${MAX_STEPS:-2}; SAVE_STEPS=${SAVE_STEPS:-2}; SAVE_START_STEP=${SAVE_START_STEP:-0}
else MAX_STEPS=${MAX_STEPS:-30000}; SAVE_STEPS=${SAVE_STEPS:-5000}; SAVE_START_STEP=${SAVE_START_STEP:-15000}; fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl2b_siglip2_da3_4layer_libero_cur_k1}

MODEL_PATH=$W2B/Qwen3-VL-2B-Instruct
SIGLIP2_MODEL_DIR=${SIGLIP2_MODEL_DIR:-$W2B/siglip2-large-patch16-256}
DA3_CKPT_DIR=$W2B/DA3-LARGE-1.1
DA3_CODE_ROOT=$J/vlm4wam_2b/code/Depth-Anything-3

for p in "$PY" "$MODEL_PATH" "$SIGLIP2_MODEL_DIR/config.json" "$DA3_CKPT_DIR/config.json" \
  "$DA3_CODE_ROOT/src/depth_anything_3/api.py" \
  "$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" "$ROOT/data/libero_qwen" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" "$DATA_ROOT/libero_10_no_noops_lerobot"; do
  [[ -e "$p" ]] || { echo "ERROR missing: $p" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs" "$ROOT/cache/triton" "$ROOT/cache/inductor"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PLANNER_WANDB=${PLANNER_WANDB:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$ROOT/cache/triton TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "$REPO_ROOT"
exec env \
  NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
  EXPECTED_GLOBAL_BATCH="${EXPECTED_GLOBAL_BATCH:-256}" MAX_STEPS="$MAX_STEPS" \
  SAVE_STEPS="$SAVE_STEPS" SAVE_START_STEP="$SAVE_START_STEP" \
  FULL_FINETUNE=1 NUM_WORKERS="${NUM_WORKERS:-8}" \
  LR="${LR:-4.24e-5}" HEAD_LR="${HEAD_LR:-4.24e-4}" WARMUP_STEPS="${WARMUP_STEPS:-2500}" \
  PY="$PY" WEIGHTS="$W2B" MODEL_PATH="$MODEL_PATH" \
  HEAD_WARMSTART_CKPT="" SEMANTIC_DIM=0 \
  VIDEO_TARGET_TYPE=siglip2 DEPTH_TARGET_TYPE=da3 \
  SIGLIP2_MODEL_DIR="$SIGLIP2_MODEL_DIR" SIGLIP2_GRID_SIZE="${SIGLIP2_GRID_SIZE:-16}" \
  SIGLIP2_INPUT_SIZE="${SIGLIP2_INPUT_SIZE:-256}" \
  DA3_CKPT_DIR="$DA3_CKPT_DIR" DA3_CODE_ROOT="$DA3_CODE_ROOT" DA3_PROCESS_RES=224 \
  DA3_ALIGN_STRATEGY=wsa_multilayer \
  DA3_TEACHER_LAYERS=11,15,19,23 DA3_LAYER_WEIGHTS=1.0,1.2,1.4,1.6 \
  USE_CURRENT_ALIGNMENT=1 INDEPENDENT_MODALITY_TASK_TOKENS=1 \
  NUM_KEYFRAMES="${NUM_KEYFRAMES:-1}" KEYFRAME_SCHEME="${KEYFRAME_SCHEME:-even_future}" \
  LINGBOT_SRC_ROOT="$ROOT/code/lingbot-vla-v2" UTILS3D_MOGE_PATH="$ROOT/py_deps/utils3d_moge" \
  FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  FASTWAM_DATASET_DIRS="$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot" \
  FASTWAM_FRAME_CACHE_DIR="${FASTWAM_FRAME_CACHE_DIR:-$ROOT/data/frame_cache/libero}" \
  FASTWAM_TEXT_EMBEDDING_CACHE_DIR="$ROOT/data/libero_qwen" \
  FASTWAM_PRETRAINED_NORM_STATS="$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
```

- [ ] **Step 3: Make the launcher executable**

Run:

```bash
chmod 775 qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3_4layer.sh
```

Expected: exit status `0`.

- [ ] **Step 4: Run the static contract and verify it passes**

Run:

```bash
launcher=qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3_4layer.sh
baseline=qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3.sh
test -f "$launcher"
test -x "$launcher"
bash -n "$launcher"
case "$(basename "$launcher")" in *wsa*) exit 1;; esac
rg -q 'OUTPUT_DIR=.*qwen3vl2b_siglip2_da3_4layer_libero_cur_k1' "$launcher"
rg -q 'DA3_ALIGN_STRATEGY=wsa_multilayer' "$launcher"
rg -q 'DA3_TEACHER_LAYERS=11,15,19,23' "$launcher"
rg -q 'DA3_LAYER_WEIGHTS=1.0,1.2,1.4,1.6' "$launcher"
test "$(sha256sum "$baseline" | cut -d' ' -f1)" = '9d9e05b46ab7868684b02f465e0829285a40916580e5ed13d9ef114c7fbbc9f0'
rg -q -- '--da3-teacher-layers.*DA3_TEACHER_LAYERS.*--da3-layer-weights.*DA3_LAYER_WEIGHTS' qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
git diff --check
```

Expected: every command exits `0` with no syntax or whitespace errors.

- [ ] **Step 5: Review the exact scope**

Run:

```bash
git status --short
git diff -- qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3_4layer.sh
```

Expected: the implementation diff contains only the new four-layer launcher; the previously committed design and plan documents may appear in Git history but not as uncommitted changes.

- [ ] **Step 6: Commit the launcher**

Run:

```bash
git add qwen3_vl_semantic_planner/dinov3_da3_2b/launch_pod_2b_siglip2da3_4layer.sh
git commit -m "feat: add SigLIP2 DA3 four-layer launcher"
```

Expected: one commit creating the executable launcher.
