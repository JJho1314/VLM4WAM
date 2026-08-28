### Task 4: Make one generic launcher and two machine profiles

**Files:**
- Modify: `tests/test_lingbot_zero2_runtime.py`
- Modify: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh`
- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh`
- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch`

**Interfaces:**
- Generic launcher consumes machine paths and hyperparameters through environment variables.
- Profiles produce complete machine-specific environments and call only `train_lingbot_dino_4b.sh`.

- [ ] **Step 1: Add a failing canonical-launcher contract test**

Append:

```python
GENERIC_LAUNCHER = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
    / "train_lingbot_dino_4b.sh"
)
POD_LAUNCHER = GENERIC_LAUNCHER.with_name("train_lingbot_fastwam_pod.sh")
HPC_LAUNCHER = GENERIC_LAUNCHER.with_name("train_lingbot_fastwam_hpc3.sbatch")


def test_only_canonical_launchers_are_referenced():
    generic = GENERIC_LAUNCHER.read_text(encoding="utf-8")
    pod = POD_LAUNCHER.read_text(encoding="utf-8")
    hpc = HPC_LAUNCHER.read_text(encoding="utf-8")
    assert "make_zero2_config.py" in generic
    assert "--expected-global-batch" in generic
    assert "NUM_TASK_TOKENS=${NUM_TASK_TOKENS:-64}" in generic
    assert "BATCH_SIZE=${BATCH_SIZE:-8}" in generic
    assert "GRAD_ACCUM=${GRAD_ACCUM:-2}" in generic
    assert "train_lingbot_dino_4b.sh" in pod
    assert "train_lingbot_dino_4b.sh" in hpc
    assert "train_lingbot_current_future_fastwam_k1.sh" not in pod + hpc
```

Also update the existing `_capture_base_launcher_args` test environment in
`tests/test_lingbot_dino_depth_contract.py` so its fake Python process exercises
the direct single-process path:

```python
"USE_DEEPSPEED": "0",
"BATCH_SIZE": "1",
"GRAD_ACCUM": "1",
"EXPECTED_GLOBAL_BATCH": "1",
```

- [ ] **Step 2: Run the launcher test and verify RED**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py::test_only_canonical_launchers_are_referenced
```

Expected: FAIL because the two canonical profiles do not exist and the generic launcher still uses torchrun defaults.

- [ ] **Step 3: Update the generic launch defaults and arguments**

Set these generic defaults in `train_lingbot_dino_4b.sh`:

```bash
NUM_GPUS=${NUM_GPUS:-8}
USE_DEEPSPEED=${USE_DEEPSPEED:-1}
USE_DEPTH=${USE_DEPTH:-1}
USE_CURRENT_ALIGNMENT=${USE_CURRENT_ALIGNMENT:-1}
INDEPENDENT_MODALITY_TASK_TOKENS=${INDEPENDENT_MODALITY_TASK_TOKENS:-1}
NUM_TASK_TOKENS=${NUM_TASK_TOKENS:-64}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-9}
NUM_KEYFRAMES=${NUM_KEYFRAMES:-1}
GRID_SIZE=${GRID_SIZE:-16}
KEYFRAME_SCHEME=${KEYFRAME_SCHEME:-even_future}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
EXPECTED_GLOBAL_BATCH=${EXPECTED_GLOBAL_BATCH:-128}
MAX_STEPS=${MAX_STEPS:-12000}
LR=${LR:-3e-5}
HEAD_LR=${HEAD_LR:-3e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
```

Add this trainer argument:

```bash
--expected-global-batch "$EXPECTED_GLOBAL_BATCH"
```

Remove `--ddp-find-unused-parameters`. Replace the torchrun tail with:

```bash
CONFIG_DIR="$OUTPUT_DIR/runtime_config"
mkdir -p "$CONFIG_DIR"
if [[ "$USE_DEEPSPEED" == "1" ]]; then
  ACCELERATE_CONFIG=$(
    "$PY" "$HERE/make_zero2_config.py" \
      --grad-accum "$GRAD_ACCUM" \
      --num-processes "$NUM_GPUS" \
      --output-dir "$CONFIG_DIR"
  )
  exec "$PY" -m accelerate.commands.launch \
    --config_file "$ACCELERATE_CONFIG" \
    "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
fi
exec "$PY" "$TRAIN_SCRIPT" "${TRAIN_ARGS[@]}"
```

- [ ] **Step 4: Add the POD profile**

Create `train_lingbot_fastwam_pod.sh` as a thin profile that validates paths,
exports offline/cache settings, and invokes the generic launcher:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/nas/junjie}
REPO_ROOT=${REPO_ROOT:-$ROOT/code/VLM4WAM_k1_zero2_20260713}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/LIBERO-fastwam}
WEIGHTS=${WEIGHTS:-$ROOT/weights}
PY=${PY:-/opt/conda/envs/vlm4wam/bin/python}
RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
if [[ "$RUN_KIND" == "smoke" ]]; then
  MAX_STEPS=${MAX_STEPS:-2}
  SAVE_STEPS=${SAVE_STEPS:-2}
else
  MAX_STEPS=${MAX_STEPS:-12000}
  SAVE_STEPS=${SAVE_STEPS:-1000}
fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl4b_lingbot_independent_q64_zero2_k1_b${BATCH_SIZE}a${GRAD_ACCUM}}

for path in \
  "$PY" \
  "$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  "$WEIGHTS/lingbot_align_heads_warmstart/model.safetensors.index.json" \
  "$WEIGHTS/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
  "$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  "$WEIGHTS/moge-2-vitb-normal/model.pt" \
  "$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" \
  "$ROOT/data/libero_qwen" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" \
  "$DATA_ROOT/libero_object_no_noops_lerobot" \
  "$DATA_ROOT/libero_goal_no_noops_lerobot" \
  "$DATA_ROOT/libero_10_no_noops_lerobot"; do
  [[ -e "$path" ]] || { echo "ERROR: missing required path: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs" "$ROOT/cache/triton" "$ROOT/cache/inductor"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PLANNER_WANDB=${PLANNER_WANDB:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$ROOT/cache/inductor
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "$REPO_ROOT"
exec env \
  NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
  EXPECTED_GLOBAL_BATCH=128 MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  FULL_FINETUNE=1 NUM_WORKERS=4 LR=3e-5 HEAD_LR=3e-4 WARMUP_STEPS=1000 \
  PY="$PY" WEIGHTS="$WEIGHTS" \
  MODEL_PATH="$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  LINGBOT_6B="$WEIGHTS/lingbot-vla-v2-6b" \
  HEAD_WARMSTART_CKPT="$WEIGHTS/lingbot_align_heads_warmstart" \
  DEPTH_MOGE_PATH="$WEIGHTS/moge-2-vitb-normal/model.pt" \
  DEPTH_MORGBD_PATH="$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  LINGBOT_SRC_ROOT="$ROOT/code/lingbot-vla-v2" \
  UTILS3D_MOGE_PATH="$ROOT/py_deps/utils3d_moge" \
  FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  FASTWAM_DATASET_DIRS="$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot" \
  FASTWAM_TEXT_EMBEDDING_CACHE_DIR="$ROOT/data/libero_qwen" \
  FASTWAM_PRETRAINED_NORM_STATS="$ROOT/data/LIBERO-fastwam_meta/dataset_stats.json" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
```

- [ ] **Step 5: Add the HPC3 profile**

Create `train_lingbot_fastwam_hpc3.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH -J vlmp_zero2_q64
#SBATCH -p acd_u
#SBATCH --gres=gpu:8
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 2-00:00:00
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

WORKSPACE=${WORKSPACE:-/data/user/jhe724/workspace}
REPO_ROOT=${REPO_ROOT:-$WORKSPACE/VLM4WAM_lingbot_zero2_20260713}
DATA_ROOT=${DATA_ROOT:-$WORKSPACE/datasets/LIBERO-fastwam}
TEXT_CACHE=${TEXT_CACHE:-$WORKSPACE/datasets/libero_qwen}
NORM_STATS=${NORM_STATS:-$WORKSPACE/datasets/LIBERO-fastwam_meta/dataset_stats.json}
WEIGHTS=${WEIGHTS:-$WORKSPACE/weights}
PY=${PY:-$HOME/.conda/envs/starVLA/bin/python}
RUN_KIND=${RUN_KIND:-formal}
NUM_GPUS=8
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
if [[ "$RUN_KIND" == "smoke" ]]; then
  MAX_STEPS=${MAX_STEPS:-2}
  SAVE_STEPS=${SAVE_STEPS:-2}
else
  MAX_STEPS=${MAX_STEPS:-12000}
  SAVE_STEPS=${SAVE_STEPS:-1000}
fi
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/qwen3vl4b_lingbot_independent_q64_zero2_k1_b${BATCH_SIZE}a${GRAD_ACCUM}_${SLURM_JOB_ID}}

for path in \
  "$PY" \
  "$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  "$WEIGHTS/lingbot_align_heads_warmstart/model.safetensors.index.json" \
  "$WEIGHTS/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth" \
  "$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  "$WEIGHTS/moge-2-vitb-normal/model.pt" \
  "$NORM_STATS" \
  "$TEXT_CACHE" \
  "$DATA_ROOT/libero_spatial_no_noops_lerobot" \
  "$DATA_ROOT/libero_object_no_noops_lerobot" \
  "$DATA_ROOT/libero_goal_no_noops_lerobot" \
  "$DATA_ROOT/libero_10_no_noops_lerobot"; do
  [[ -e "$path" ]] || { echo "ERROR: missing required path: $path" >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/logs" "$WORKSPACE/.cache/triton" "$WORKSPACE/.cache/inductor"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PLANNER_WANDB=${PLANNER_WANDB:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=$WORKSPACE/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$WORKSPACE/.cache/inductor
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "$REPO_ROOT"
exec env \
  NUM_GPUS="$NUM_GPUS" BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
  EXPECTED_GLOBAL_BATCH=128 MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
  FULL_FINETUNE=1 NUM_WORKERS=4 LR=3e-5 HEAD_LR=3e-4 WARMUP_STEPS=1000 \
  PY="$PY" WEIGHTS="$WEIGHTS" \
  MODEL_PATH="$WEIGHTS/Qwen3-VL-4B-lingbot-vlm" \
  LINGBOT_6B="$WEIGHTS/lingbot-vla-v2-6b" \
  HEAD_WARMSTART_CKPT="$WEIGHTS/lingbot_align_heads_warmstart" \
  DEPTH_MOGE_PATH="$WEIGHTS/moge-2-vitb-normal/model.pt" \
  DEPTH_MORGBD_PATH="$WEIGHTS/lingbot-vla-v2-6b/depth/model.pt" \
  LINGBOT_SRC_ROOT="$WORKSPACE/lingbot-vla-v2" \
  UTILS3D_MOGE_PATH="$WORKSPACE/py_deps/utils3d_moge" \
  FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  FASTWAM_DATASET_DIRS="$DATA_ROOT/libero_spatial_no_noops_lerobot:$DATA_ROOT/libero_object_no_noops_lerobot:$DATA_ROOT/libero_goal_no_noops_lerobot:$DATA_ROOT/libero_10_no_noops_lerobot" \
  FASTWAM_TEXT_EMBEDDING_CACHE_DIR="$TEXT_CACHE" \
  FASTWAM_PRETRAINED_NORM_STATS="$NORM_STATS" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
```

- [ ] **Step 6: Verify launcher tests and shell syntax**

Run:

```bash
pytest -q tests/test_lingbot_zero2_runtime.py::test_only_canonical_launchers_are_referenced
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch
```

Expected: the test passes and all three shell checks produce no output.

---

