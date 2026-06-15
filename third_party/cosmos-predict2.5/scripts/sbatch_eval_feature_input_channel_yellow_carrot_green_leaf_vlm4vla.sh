#!/usr/bin/env bash
# Evaluate the feature-input-channel Cosmos model on the yellow carrot with
# green leaves example. Cosmos inference receives target_feature only; target
# mask is removed before denoising.

#SBATCH --job-name=cosmos-yc-feat
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4VLA/slurm-yc-feature-input-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4VLA/slurm-yc-feature-input-%j.err

set -uo pipefail

VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jhe724/workspace/VLM4VLA}
REPO_ROOT=${REPO_ROOT:-$VLM4VLA_ROOT/third_party/cosmos-predict2.5}
DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/outputs/tavid_generation_runs/robointer_74616_yellow_carrot_prompt_targetaware_dataset}
FEATURE_DIR_NAME=${FEATURE_DIR_NAME:-target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt}
FEATURE_DIR=${FEATURE_DIR:-$DATASET_DIR/$FEATURE_DIR_NAME}
RUN_ROOT=${RUN_ROOT:-$VLM4VLA_ROOT/outputs/eval_feature_input_channel_iter2000_yellow_carrot_green_leaf_$(date +%Y%m%d_%H%M%S)}

CHECKPOINT=${CHECKPOINT:-$VLM4VLA_ROOT/outputs/droid_success_v21_feature_input_channel_stage2_lora_scene_cap200_tasktarget/cosmos_predict_v2p5/video2world/2b_droid_success_v21_feature_input_channel_stage2_lora_scene_cap200_tasktarget_49f_s234_bs2accum8_gbs128_2000step_val400_from_base/checkpoints/iter_000002000}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_feature_input_channel_target}
SEED=${SEED:-20260613}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
FPS=${FPS:-8}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-feature_input_channel_iter2000_yellow_carrot_green_leaf}

cd "$REPO_ROOT" || exit 2
module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

mkdir -p "$FEATURE_DIR" "$RUN_ROOT/logs"

echo "=== yellow carrot green-leaf feature precompute ===" | tee "$RUN_ROOT/logs/00_run_info.log"
{
  date
  hostname
  echo "repo=$REPO_ROOT"
  echo "dataset=$DATASET_DIR"
  echo "feature_dir=$FEATURE_DIR"
  echo "checkpoint=$CHECKPOINT"
  echo "experiment=$EXPERIMENT"
	  echo "run_root=$RUN_ROOT"
	  echo "seed=$SEED num_steps=$NUM_STEPS guidance=$GUIDANCE fps=$FPS"
	  echo "output_prefix=$OUTPUT_PREFIX"
	  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
	} | tee -a "$RUN_ROOT/logs/00_run_info.log"

if [ ! -f "$DATASET_DIR/videos/74616_exterior_image_1_left.mp4" ]; then
  echo "Missing yellow-carrot dataset at $DATASET_DIR" >&2
  exit 3
fi

ISAM_ENV=${ISAM_ENV:-/data/user/jhe724/.conda/envs/instructsam}
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
  --output-dir-name "$FEATURE_DIR_NAME" \
  --query-template "Please segment the yellow carrot with green leaves in the image." \
  --fallback-query "Please segment the yellow carrot with green leaves in the image." \
  --feature-mode decoder_dense \
  --expected-feature-dim 256 \
  --combine-mode best \
  --overwrite \
  --limit 1 \
  --log-every 1 \
  2>&1 | tee "$RUN_ROOT/logs/01_precompute_decoder_dense_green_leaf.log"
precompute_status=${PIPESTATUS[0]}
if [ "$precompute_status" -ne 0 ]; then
  exit "$precompute_status"
fi

python - <<PY | tee "$RUN_ROOT/logs/02_feature_shape.log"
import torch
from pathlib import Path
p = Path("$FEATURE_DIR/74616_exterior_image_1_left.pt")
obj = torch.load(p, map_location="cpu", weights_only=False)
feat = obj["target_feature"] if isinstance(obj, dict) else obj
print("feature_path:", p)
print("feature_shape:", tuple(feat.shape))
print("feature_mode:", obj.get("feature_mode") if isinstance(obj, dict) else None)
print("query:", obj.get("query") if isinstance(obj, dict) else None)
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
    dataloader_train.dataset.target_feature_dir="$FEATURE_DIR" \
    dataloader_train.dataset.target_feature_default_to_zero=False \
    trainer.grad_accum_iter=1 \
    trainer.run_validation=False \
    2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "$status" > "$RUN_ROOT/logs/${name}.exit"
  return "$status"
}

overall=0
run_variant keep keep || overall=1
run_variant zero zero || overall=1
run_variant drop drop || overall=1

if [ "$overall" -eq 0 ]; then
  python scripts/analyze_target_feature_ablation.py \
    --run-root "$RUN_ROOT" \
	    --variants keep zero drop \
	    --mask-npz "$DATASET_DIR/target_masks/74616_exterior_image_1_left.npz" \
	    --output-prefix "$OUTPUT_PREFIX" \
	    2>&1 | tee "$RUN_ROOT/logs/06_analyze.log" || overall=1
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
    "feature_dir": "$FEATURE_DIR",
    "seed": int("$SEED"),
    "num_steps": int("$NUM_STEPS"),
    "guidance": float("$GUIDANCE"),
    "variants": ["keep", "zero", "drop"],
	    "cosmos_target_mask_removed": True,
	    "overall_exit": int("$overall"),
	    "contact_sheet": str(root / ("$OUTPUT_PREFIX" + "_contact_sheet.jpg")),
	    "diff_sheet": str(root / ("$OUTPUT_PREFIX" + "_diff_vs_keep.jpg")),
	    "metrics": str(root / ("$OUTPUT_PREFIX" + "_metrics.json")),
	}
(root / "eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

exit "$overall"
