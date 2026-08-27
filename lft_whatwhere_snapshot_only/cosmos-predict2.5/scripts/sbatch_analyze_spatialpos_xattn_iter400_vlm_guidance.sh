#!/usr/bin/env bash
# Wait for iter_000000400 of the spatial-position target cross-attention run,
# then run mask-free feature-intervention generations plus intermediate
# attention/matching/velocity diagnostics.

#SBATCH --job-name=mgv3ctx400diag
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=18:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-mgv3ctx400diag-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-mgv3ctx400diag-%j.err

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

export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_v21_match_ground_v3_target_context_spatialpos_crossattn_49f_s123_vlm4wam}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_match_ground_v3_target_context_spatialpos_crossattn}
JOB_NAME=${JOB_NAME:-2b_mgv3_target_context_spatialpos_xattn_iou50_49f_s123_bs1accum16_gbs128_2000}
RUN_DIR="$IMAGINAIRE_OUTPUT_ROOT/cosmos_predict_v2p5/video2world/$JOB_NAME"
CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoints/iter_000000400}

BASE_DS=${BASE_DS:-$VLM4WAM_ROOT/eval_prev_iter2000_full/input_datasets}
CARROT_DS=${CARROT_DS:-$BASE_DS/robointer_74616_yellow_carrot_prompt_targetaware_dataset}
BANANA_DS=${BANANA_DS:-$BASE_DS/robointer_74616_banana_prompt_targetaware_dataset}
STEM=${STEM:-74616_exterior_image_1_left}

CARROT_RAW=${CARROT_RAW:-$CARROT_DS/target_features_rawseg_ft/$STEM.pt}
CARROT_DENSE=${CARROT_DENSE:-$CARROT_DS/target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt_s20260613/$STEM.pt}
BANANA_RAW=${BANANA_RAW:-$BANANA_DS/target_features_rawseg_stage2_lora_banana_prompt_s20260613/$STEM.pt}
BANANA_DENSE=${BANANA_DENSE:-$BANANA_DS/target_features_instructsam_decoder_dense_stage2_lora_banana_prompt_s20260613/$STEM.pt}
MASK_NPZ=${MASK_NPZ:-$CARROT_DS/target_masks/$STEM.npz}

SEED=${SEED:-20260613}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
FPS=${FPS:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
TAVID_ATTN_QUERY_CHUNK_SIZE=${TAVID_ATTN_QUERY_CHUNK_SIZE:-1024}
WAIT_FOR_CHECKPOINT_SECONDS=${WAIT_FOR_CHECKPOINT_SECONDS:-43200}
WAIT_POLL_SECONDS=${WAIT_POLL_SECONDS:-120}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=${RUN_ROOT:-$VLM4WAM_ROOT/feature_guidance_analysis/spatialpos_xattn_iter400_yellow_carrot_${TIMESTAMP}}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-feature_guidance_iter400}

export DROID_SUCCESS_V21_TAVID_DIR=$CARROT_DS
export DROID_SUCCESS_V21_TAVID_VAL_DIR=$CARROT_DS
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-1}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

waited=0
while [ ! -e "$CHECKPOINT" ]; do
  if [ "$waited" -ge "$WAIT_FOR_CHECKPOINT_SECONDS" ]; then
    echo "Timed out waiting for checkpoint: $CHECKPOINT" >&2
    exit 3
  fi
  echo "Waiting for checkpoint: $CHECKPOINT (${waited}s/${WAIT_FOR_CHECKPOINT_SECONDS}s)"
  sleep "$WAIT_POLL_SECONDS"
  waited=$((waited + WAIT_POLL_SECONDS))
done

missing=0
for path in "$CARROT_DS/videos/$STEM.mp4" "$CARROT_DS/metas/$STEM.txt" "$CARROT_RAW" "$CARROT_DENSE" "$BANANA_RAW" "$BANANA_DENSE" "$MASK_NPZ"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  exit 2
fi

mkdir -p "$RUN_ROOT/logs"

{
  date
  hostname
  echo "repo=$REPO_ROOT"
  echo "run_root=$RUN_ROOT"
  echo "checkpoint=$CHECKPOINT"
  echo "experiment=$EXPERIMENT"
  echo "job_name=$JOB_NAME"
  echo "carrot_ds=$CARROT_DS"
  echo "carrot_raw=$CARROT_RAW"
  echo "carrot_dense=$CARROT_DENSE"
  echo "banana_raw=$BANANA_RAW"
  echo "banana_dense=$BANANA_DENSE"
  echo "seed=$SEED num_steps=$NUM_STEPS guidance=$GUIDANCE fps=$FPS"
  nvidia-smi -L
  sha256sum "$CARROT_RAW" "$CARROT_DENSE" "$BANANA_RAW" "$BANANA_DENSE" 2>/dev/null || true
} | tee "$RUN_ROOT/logs/00_run_info.log"

python - <<PY | tee "$RUN_ROOT/logs/01_feature_check.log"
import json, torch
from pathlib import Path
paths = {
    "carrot_raw": Path("$CARROT_RAW"),
    "carrot_dense": Path("$CARROT_DENSE"),
    "banana_raw": Path("$BANANA_RAW"),
    "banana_dense": Path("$BANANA_DENSE"),
}
feats = {}
for name, path in paths.items():
    obj = torch.load(path, map_location="cpu", weights_only=False)
    feat = obj["target_feature"] if isinstance(obj, dict) else obj
    feats[name] = feat.float()
    print(json.dumps({
        "name": name,
        "path": str(path),
        "shape": list(feat.shape),
        "feature_mode": obj.get("feature_mode") if isinstance(obj, dict) else None,
        "query": obj.get("query") if isinstance(obj, dict) else None,
        "mean": float(feat.float().mean()),
        "std": float(feat.float().std()),
    }, ensure_ascii=False))
for left, right in [("carrot_raw", "banana_raw"), ("carrot_dense", "banana_dense")]:
    a, b = feats[left].reshape(-1), feats[right].reshape(-1)
    n = min(a.numel(), b.numel())
    print(json.dumps({
        "pair": f"{left}_vs_{right}",
        "cosine": float(torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=0)),
        "relative_l2": float(torch.linalg.vector_norm(a[:n] - b[:n]) / torch.linalg.vector_norm(a[:n]).clamp_min(1e-6)),
    }, ensure_ascii=False))
PY

run_variant() {
  local name="$1"
  local mode="$2"
  local raw_path="${3:-}"
  local dense_path="${4:-}"
  local out="$RUN_ROOT/$name"
  local log="$RUN_ROOT/logs/${name}.log"
  mkdir -p "$out"
  local extra=()
  if [ "$mode" = "path" ]; then
    extra=(--target-feature-path "$raw_path" --target-dense-feature-path "$dense_path")
  fi

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
    "${extra[@]}" \
    -- experiment="$EXPERIMENT" \
    job.name="$JOB_NAME" \
    dataloader_train.batch_size="$BATCH_SIZE" \
    dataloader_train.num_workers="$NUM_WORKERS" \
    dataloader_train.drop_last=False \
    dataloader_train.dataset.target_mask_dir=none \
    dataloader_train.dataset.target_mask_default_to_zero=True \
    dataloader_train.dataset.target_feature_dir="$(dirname "$CARROT_RAW")" \
    dataloader_train.dataset.target_feature_default_to_zero=False \
    dataloader_train.dataset.target_feature_dim=2048 \
    dataloader_train.dataset.target_feature_max_tokens=16 \
    dataloader_train.dataset.target_dense_feature_dir="$(dirname "$CARROT_DENSE")" \
    dataloader_train.dataset.target_dense_feature_default_to_zero=False \
    dataloader_train.dataset.target_dense_feature_dim=256 \
    dataloader_train.dataset.target_dense_feature_max_tokens=1024 \
    trainer.grad_accum_iter=1 \
    trainer.run_validation=False \
    model.config.net.tavid_attn_query_chunk_size="$TAVID_ATTN_QUERY_CHUNK_SIZE" \
    2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "$status" > "$RUN_ROOT/logs/${name}.exit"
  return "$status"
}

overall=0
run_variant keep path "$CARROT_RAW" "$CARROT_DENSE" || overall=1
run_variant wrong_banana path "$BANANA_RAW" "$BANANA_DENSE" || overall=1
run_variant zero zero || overall=1
run_variant drop drop || overall=1

if [ "$overall" -eq 0 ]; then
  python scripts/analyze_target_feature_ablation.py \
    --run-root "$RUN_ROOT" \
    --variants keep wrong_banana zero drop \
    --mask-npz "$MASK_NPZ" \
    --output-prefix "$OUTPUT_PREFIX" \
    2>&1 | tee "$RUN_ROOT/logs/06_generation_ablation_analysis.log" || overall=1
fi

mkdir -p "$RUN_ROOT/intermediate"
torchrun --standalone --nproc_per_node=1 -m scripts.analyze_vlm_feature_guidance_intermediate \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint="$CHECKPOINT" \
  --output-dir="$RUN_ROOT/intermediate" \
  --split=train \
  --num-samples=1 \
  --max-batches=1 \
  --num-conditional-frames=1 \
  --seed="$SEED" \
  --blocks=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
  --selected-blocks=4,8,12,16,20,24 \
  --variants keep zero drop wrong \
  --wrong-target-feature-path="$BANANA_RAW" \
  --wrong-target-dense-feature-path="$BANANA_DENSE" \
  --token-source=feature \
  -- experiment="$EXPERIMENT" \
  job.name="$JOB_NAME" \
  dataloader_train.batch_size="$BATCH_SIZE" \
  dataloader_train.num_workers="$NUM_WORKERS" \
  dataloader_train.drop_last=False \
  dataloader_train.dataset.dataset_dir="$CARROT_DS" \
  dataloader_train.sampler.dataset.dataset_dir="$CARROT_DS" \
  dataloader_train.dataset.target_mask_dir=auto \
  dataloader_train.dataset.target_feature_dir="$(dirname "$CARROT_RAW")" \
  dataloader_train.dataset.target_feature_default_to_zero=False \
  dataloader_train.dataset.target_feature_dim=2048 \
  dataloader_train.dataset.target_feature_max_tokens=16 \
  dataloader_train.dataset.target_dense_feature_dir="$(dirname "$CARROT_DENSE")" \
  dataloader_train.dataset.target_dense_feature_default_to_zero=False \
  dataloader_train.dataset.target_dense_feature_dim=256 \
  dataloader_train.dataset.target_dense_feature_max_tokens=1024 \
  model.config.net.tavid_attn_query_chunk_size="$TAVID_ATTN_QUERY_CHUNK_SIZE" \
  2>&1 | tee "$RUN_ROOT/logs/07_intermediate_guidance.log" || overall=1

python - <<PY
import json
from pathlib import Path
root = Path("$RUN_ROOT")
summary = {
    "run_root": str(root),
    "checkpoint": "$CHECKPOINT",
    "experiment": "$EXPERIMENT",
    "job_name": "$JOB_NAME",
    "explicit_mask_used_for_generation": False,
    "generation_variants": ["keep", "wrong_banana", "zero", "drop"],
    "intermediate_note": "GT mask is used only inside the intermediate analysis script to compute metrics and overlays.",
    "generation_contact_sheet": str(root / "${OUTPUT_PREFIX}_contact_sheet.jpg"),
    "generation_diff_sheet": str(root / "${OUTPUT_PREFIX}_diff_vs_keep.jpg"),
    "generation_metrics": str(root / "${OUTPUT_PREFIX}_metrics.json"),
    "intermediate_summary": str(root / "intermediate/vlm_feature_guidance_intermediate_summary.json"),
    "intermediate_figure": str(root / "intermediate/sample_000_vlm_guidance_intermediate.jpg"),
    "overall_exit": int("$overall"),
}
(root / "${OUTPUT_PREFIX}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "run_root=$RUN_ROOT"
exit "$overall"
