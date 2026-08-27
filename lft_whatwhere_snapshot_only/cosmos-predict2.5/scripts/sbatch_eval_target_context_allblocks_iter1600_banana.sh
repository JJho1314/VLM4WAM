#!/usr/bin/env bash
# Evaluate the current mask-free all-block target-context checkpoint on the
# original yellow-banana prompt. Inference removes target masks and compares
# keep/zero/drop InstructSAM target features.

#SBATCH --job-name=banana-mgv3ctx
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-eval-banana-mgv3ctx-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-eval-banana-mgv3ctx-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
REPO_ROOT=${REPO_ROOT:-$VLM4WAM_ROOT/third_party/cosmos-predict2.5}
DATASET_DIR=${DATASET_DIR:-$VLM4WAM_ROOT/eval_prev_iter2000_full/input_datasets/robointer_74616_banana_prompt_targetaware_dataset}
RAW_FEATURE_DIR_NAME=${RAW_FEATURE_DIR_NAME:-target_features_rawseg_stage2_lora_banana_prompt_s20260613}
DENSE_FEATURE_DIR_NAME=${DENSE_FEATURE_DIR_NAME:-target_features_instructsam_decoder_dense_stage2_lora_banana_prompt_s20260613}
RAW_FEATURE_DIR=${RAW_FEATURE_DIR:-$DATASET_DIR/$RAW_FEATURE_DIR_NAME}
DENSE_FEATURE_DIR=${DENSE_FEATURE_DIR:-$DATASET_DIR/$DENSE_FEATURE_DIR_NAME}

RUN_DIR=${RUN_DIR:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_v21_match_ground_v3_target_context_allblocks_49f_s123_vlm4wam_fix2/cosmos_predict_v2p5/video2world/2b_mgv3_target_context_allblocks_iou50_49f_s123_bs1accum16_gbs128_1600_fix2}
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoints/iter_000001600}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_match_ground_v3_target_context_allblocks}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/outputs/eval_target_context_allblocks_iter1600_banana_$(date +%Y%m%d_%H%M%S)}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-target_context_allblocks_iter1600_banana}
SEED=${SEED:-20260613}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
FPS=${FPS:-8}
BANANA_PROMPT=${BANANA_PROMPT:-A Franka robotic arm with a parallel-jaw gripper put the [TGT] yellow banana into the black pot on the counter.}
BANANA_QUERY=${BANANA_QUERY:-Please segment the yellow banana in the image.}

mkdir -p "$VLM4WAM_ROOT/logs" "$RUN_ROOT/logs"

if [ ! -d "$REPO_ROOT" ]; then
  echo "Missing repo: $REPO_ROOT" >&2
  exit 2
fi
if [ ! -f "$DATASET_DIR/videos/74616_exterior_image_1_left.mp4" ]; then
  echo "Missing banana video under $DATASET_DIR/videos" >&2
  exit 3
fi
if [ ! -d "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 4
fi

cd "$REPO_ROOT" || exit 2
module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true
export PATH=/data/apps/gcc/11.5/bin:${PATH}
export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

mkdir -p "$DATASET_DIR/metas"
printf '%s\n' "$BANANA_PROMPT" > "$DATASET_DIR/metas/74616_exterior_image_1_left.txt"

{
  date
  hostname
  echo "repo=$REPO_ROOT"
  echo "dataset=$DATASET_DIR"
  echo "prompt=$BANANA_PROMPT"
  echo "query=$BANANA_QUERY"
  echo "raw_feature_dir=$RAW_FEATURE_DIR"
  echo "dense_feature_dir=$DENSE_FEATURE_DIR"
  echo "checkpoint=$CHECKPOINT"
  echo "experiment=$EXPERIMENT"
  echo "run_root=$RUN_ROOT"
  echo "seed=$SEED num_steps=$NUM_STEPS guidance=$GUIDANCE fps=$FPS"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
} | tee "$RUN_ROOT/logs/00_run_info.log"

ISAM_ENV=${ISAM_ENV:-/data/user/jhe724/.conda/envs/instructsam}
export VIRTUAL_ENV=$ISAM_ENV
export PATH=/data/apps/gcc/11.5/bin:$ISAM_ENV/bin:$PATH
unset PYTHONHOME
export HF_HUB_OFFLINE=1
export INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/InstructSAM}
export INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/InstructSAM/work_dirs/instructsam_stage2_complete_lora}
export PYTHONPATH="$REPO_ROOT/scripts/_env_stubs:$REPO_ROOT:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export COSMOS_SKIP_CUDA_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export INSTRUCTSAM_DECODER_DENSE_SIZE=${INSTRUCTSAM_DECODER_DENSE_SIZE:-32}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/precompute_instructsam_target_features.py \
  --dataset-dir "$DATASET_DIR" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --output-dir-name "$RAW_FEATURE_DIR_NAME" \
  --query-template "$BANANA_QUERY" \
  --fallback-query "$BANANA_QUERY" \
  --feature-mode raw_seg \
  --expected-feature-dim 2048 \
  --combine-mode best \
  --overwrite \
  --limit 1 \
  --log-every 1 \
  2>&1 | tee "$RUN_ROOT/logs/01_precompute_rawseg_banana.log"
precompute_raw_status=${PIPESTATUS[0]}
if [ "$precompute_raw_status" -ne 0 ]; then
  exit "$precompute_raw_status"
fi

python scripts/precompute_instructsam_target_features.py \
  --dataset-dir "$DATASET_DIR" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --output-dir-name "$DENSE_FEATURE_DIR_NAME" \
  --query-template "$BANANA_QUERY" \
  --fallback-query "$BANANA_QUERY" \
  --feature-mode decoder_dense \
  --expected-feature-dim 256 \
  --combine-mode best \
  --overwrite \
  --limit 1 \
  --log-every 1 \
  2>&1 | tee "$RUN_ROOT/logs/02_precompute_decoder_dense_banana.log"
precompute_dense_status=${PIPESTATUS[0]}
if [ "$precompute_dense_status" -ne 0 ]; then
  exit "$precompute_dense_status"
fi

python - <<PY | tee "$RUN_ROOT/logs/03_feature_shape.log"
import json
from pathlib import Path
import torch
stem = "74616_exterior_image_1_left"
for name, path, dim in (
    ("raw_feature", Path("$RAW_FEATURE_DIR") / f"{stem}.pt", 2048),
    ("dense_feature", Path("$DENSE_FEATURE_DIR") / f"{stem}.pt", 256),
):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    feat = obj["target_feature"] if isinstance(obj, dict) else obj
    info = {
        "name": name,
        "path": str(path),
        "shape": list(feat.shape),
        "expected_dim": dim,
        "query": obj.get("query") if isinstance(obj, dict) else None,
        "feature_mode": obj.get("feature_mode") if isinstance(obj, dict) else None,
        "mean": float(feat.float().mean()),
        "std": float(feat.float().std()),
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if feat.shape[-1] != dim:
        raise SystemExit(f"{name}: expected dim {dim}, got {feat.shape[-1]}")
PY

COSMOS_VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV="$COSMOS_VENV"
export PATH=/data/apps/gcc/11.5/bin:$COSMOS_VENV/bin:$PATH
unset PYTHONHOME
NV_LIB=$COSMOS_VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-cuda:$REPO_ROOT/packages/cosmos-oss:${PYTHONPATH:-}"
export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/user/jhe724/workspace/weights}
export WANDB_MODE=disabled
export DROID_SUCCESS_V21_TAVID_DIR="$DATASET_DIR"
export DROID_SUCCESS_V21_TAVID_VAL_DIR="$DATASET_DIR"
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-1}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

run_variant() {
  local name=$1
  local mode=$2
  local out="$RUN_ROOT/$name"
  local log="$RUN_ROOT/logs/${name}.log"
  mkdir -p "$out"
  set +e
  torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
    --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$out" \
    --num-samples 1 \
    --num-steps "$NUM_STEPS" \
    --guidance "$GUIDANCE" \
    --seed "$SEED" \
    --fps "$FPS" \
    --max-batches 1 \
    --standalone-only \
    --reuse-encoded-latent \
    --offload-denoiser-during-vae \
    --offload-denoiser-before-decode \
    --allow-empty-target-mask \
    --remove-target-mask \
    --target-feature-mode "$mode" \
    -- experiment="$EXPERIMENT" \
    dataloader_train.batch_size=1 \
    dataloader_train.num_workers=2 \
    dataloader_train.drop_last=False \
    dataloader_train.dataset.target_mask_dir=none \
    dataloader_train.dataset.target_mask_default_to_zero=True \
    dataloader_train.dataset.target_feature_dir="$RAW_FEATURE_DIR" \
    dataloader_train.dataset.target_feature_default_to_zero=False \
    dataloader_train.dataset.target_feature_dim=2048 \
    dataloader_train.dataset.target_feature_max_tokens=16 \
    dataloader_train.dataset.target_dense_feature_dir="$DENSE_FEATURE_DIR" \
    dataloader_train.dataset.target_dense_feature_default_to_zero=False \
    dataloader_train.dataset.target_dense_feature_dim=256 \
    dataloader_train.dataset.target_dense_feature_max_tokens=1024 \
    trainer.grad_accum_iter=1 \
    trainer.run_validation=False \
    2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -u
  echo "$status" > "$RUN_ROOT/logs/${name}.exit"
  return "$status"
}

overall=0
run_variant keep keep || overall=1
run_variant zero zero || overall=1
run_variant drop drop || overall=1

if [ "$overall" -eq 0 ]; then
  analyze_args=(
    --run-root "$RUN_ROOT"
    --variants keep zero drop
    --output-prefix "$OUTPUT_PREFIX"
  )
  if [ -f "$DATASET_DIR/target_masks/74616_exterior_image_1_left.npz" ]; then
    analyze_args+=(--mask-npz "$DATASET_DIR/target_masks/74616_exterior_image_1_left.npz")
  fi
  python scripts/analyze_target_feature_ablation.py "${analyze_args[@]}" \
    2>&1 | tee "$RUN_ROOT/logs/07_analyze.log" || overall=1
fi

python - <<PY
import json
from pathlib import Path
root = Path("$RUN_ROOT")
summary = {
    "run_root": str(root),
    "checkpoint": "$CHECKPOINT",
    "experiment": "$EXPERIMENT",
    "dataset": "$DATASET_DIR",
    "prompt": "$BANANA_PROMPT",
    "query": "$BANANA_QUERY",
    "raw_feature_dir": "$RAW_FEATURE_DIR",
    "dense_feature_dir": "$DENSE_FEATURE_DIR",
    "seed": int("$SEED"),
    "num_steps": int("$NUM_STEPS"),
    "guidance": float("$GUIDANCE"),
    "fps": int("$FPS"),
    "variants": ["keep", "zero", "drop"],
    "cosmos_target_mask_removed": True,
    "explicit_mask_used_for_inference": False,
    "feature_ablation_applies_to": ["target_feature", "target_dense_feature"],
    "overall_exit": int("$overall"),
    "contact_sheet": str(root / ("$OUTPUT_PREFIX" + "_contact_sheet.jpg")),
    "diff_sheet": str(root / ("$OUTPUT_PREFIX" + "_diff_vs_keep.jpg")),
    "metrics": str(root / ("$OUTPUT_PREFIX" + "_metrics.json")),
}
(root / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

exit "$overall"
