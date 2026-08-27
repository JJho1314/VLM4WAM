#!/usr/bin/env bash
# Repair a SigLIP2 oracle eval run by regenerating only the wrong_other variant
# and then running the four-way ablation analysis.

#SBATCH --job-name=siglip2-oracle-repair
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-oracle-repair-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-oracle-repair-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
REPO_ROOT=${REPO_ROOT:-$VLM4WAM_ROOT/third_party/cosmos-predict2.5}
cd "$REPO_ROOT" || exit 2
mkdir -p "$VLM4WAM_ROOT/logs"

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-oss:${PYTHONPATH:-}"

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/user/jhe724/workspace/weights}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export WANDB_MODE=disabled

IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_v21_siglip2_semantic_oracle_context_320_49f_s123_vlm4wam}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_siglip2_semantic_oracle_context_320}
JOB_NAME=${JOB_NAME:-2b_siglip2_semantic_oracle_context_iou50_320_49f_s123_bs4accum4_gbs128_1600}
RUN_DIR="$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/video2world/$JOB_NAME"
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoints/iter_000001600}

BASE_DS=${BASE_DS:-$VLM4WAM_ROOT/eval_prev_iter2000_full/input_datasets}
CARROT_DS=${CARROT_DS:-$BASE_DS/robointer_74616_yellow_carrot_prompt_targetaware_dataset}
FEATURE_DIR_NAME=${FEATURE_DIR_NAME:-siglip2_semantic_plan_k6_g9_full}
WRONG_FEATURE=${WRONG_FEATURE:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half/siglip2_semantic_plan_k6_g9_full/episode_039737_left_external.pt}
STEM=${STEM:-74616_exterior_image_1_left}
MASK_NPZ=${MASK_NPZ:-$CARROT_DS/target_masks/$STEM.npz}

RUN_ROOT=${RUN_ROOT:-$VLM4WAM_ROOT/feature_guidance_analysis/siglip2_oracle_iter1600_yellow_carrot_20260626_152218}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-siglip2_oracle_iter1600_yellow_carrot}
SEED=${SEED:-20260613}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
FPS=${FPS:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
VIDEO_SIZE_H=${VIDEO_SIZE_H:-320}
VIDEO_SIZE_W=${VIDEO_SIZE_W:-576}
TAVID_ATTN_QUERY_CHUNK_SIZE=${TAVID_ATTN_QUERY_CHUNK_SIZE:-1024}

export DROID_SUCCESS_V21_TAVID_DIR=$CARROT_DS
export DROID_SUCCESS_V21_TAVID_VAL_DIR=$CARROT_DS
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-1}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/wrong_other"

missing=0
for path in "$CHECKPOINT" "$CARROT_DS/frame_ranges.json" "$CARROT_DS/videos/$STEM.mp4" "$CARROT_DS/$FEATURE_DIR_NAME/$STEM.pt" "$WRONG_FEATURE" "$MASK_NPZ"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  exit 2
fi

set +e
torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$RUN_ROOT/wrong_other" \
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
  --target-feature-mode path \
  --target-feature-path "$WRONG_FEATURE" \
  -- experiment="$EXPERIMENT" \
  job.name="$JOB_NAME" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  dataloader_train.num_workers="$NUM_WORKERS" \
  dataloader_train.drop_last=False \
  dataloader_train.dataset.video_size="[$VIDEO_SIZE_H,$VIDEO_SIZE_W]" \
  dataloader_train.sampler.dataset.video_size="[$VIDEO_SIZE_H,$VIDEO_SIZE_W]" \
  dataloader_train.dataset.dataset_dir="$CARROT_DS" \
  dataloader_train.sampler.dataset.dataset_dir="$CARROT_DS" \
  dataloader_train.dataset.target_mask_dir=none \
  dataloader_train.dataset.target_mask_default_to_zero=True \
  dataloader_train.dataset.target_feature_dir="$CARROT_DS/$FEATURE_DIR_NAME" \
  dataloader_train.sampler.dataset.target_feature_dir="$CARROT_DS/$FEATURE_DIR_NAME" \
  dataloader_train.dataset.target_feature_default_to_zero=False \
  dataloader_train.sampler.dataset.target_feature_default_to_zero=False \
  dataloader_train.dataset.target_feature_dim=1152 \
  dataloader_train.sampler.dataset.target_feature_dim=1152 \
  dataloader_train.dataset.target_feature_max_tokens=486 \
  dataloader_train.sampler.dataset.target_feature_max_tokens=486 \
  dataloader_train.dataset.target_dense_feature_dir=none \
  dataloader_train.sampler.dataset.target_dense_feature_dir=none \
  trainer.grad_accum_iter=1 \
  trainer.run_validation=False \
  model.config.net.tavid_attn_query_chunk_size="$TAVID_ATTN_QUERY_CHUNK_SIZE" \
  2>&1 | tee "$RUN_ROOT/logs/wrong_other_repair.log"
status="${PIPESTATUS[0]}"
set -u
echo "$status" > "$RUN_ROOT/logs/wrong_other_repair.exit"
if [ "$status" -ne 0 ]; then
  exit "$status"
fi

python scripts/analyze_target_feature_ablation.py \
  --run-root "$RUN_ROOT" \
  --variants keep wrong_other zero drop \
  --mask-npz "$MASK_NPZ" \
  --output-prefix "$OUTPUT_PREFIX" \
  2>&1 | tee "$RUN_ROOT/logs/06_generation_ablation_analysis.log"
status="${PIPESTATUS[0]}"
if [ "$status" -ne 0 ]; then
  exit "$status"
fi

python - <<PY
import json
from pathlib import Path
root = Path("$RUN_ROOT")
summary = {
    "run_root": str(root),
    "checkpoint": "$CHECKPOINT",
    "experiment": "$EXPERIMENT",
    "job_name": "$JOB_NAME",
    "variants": ["keep", "wrong_other", "zero", "drop"],
    "explicit_mask_used_for_generation": False,
    "video_size": [$VIDEO_SIZE_H, $VIDEO_SIZE_W],
    "num_steps": $NUM_STEPS,
    "guidance": $GUIDANCE,
    "seed": $SEED,
    "wrong_feature": "$WRONG_FEATURE",
    "contact_sheet": str(root / f"$OUTPUT_PREFIX_contact_sheet.jpg"),
    "diff_sheet": str(root / f"$OUTPUT_PREFIX_diff_sheet.jpg"),
    "metrics": str(root / f"$OUTPUT_PREFIX_metrics.json"),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
