#!/usr/bin/env bash
#SBATCH --job-name=cosmos-oracle-repro
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/oracle-repro-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/oracle-repro-%j.err
#
# Reproduce eval_results_oracle_iter3000_yellowcarrot_20260703_170049 (the "grasp the yellow carrot,
# not the banana" oracle case) on the iter3000 SG-WAM.
#
# WHAT THIS DOES
#   The original launcher (scripts/sbatch_eval_oracle_yellowcarrot.sh) had four stages:
#     1. convert the distcp checkpoint -> consolidated EMA bf16 .pt
#     2. extract the start frame + GT 49-frame clip from the source video
#     3. export the ORACLE plan: SigLIP2 over the REAL FUTURE frames  <-- this is what gets injected
#     4. run inference
#   The helper scripts for 1-3 are no longer in the repo, but everything they PRODUCE still is, so
#   this script re-runs stage 4 only and reuses those artifacts. The inference spec is copied
#   verbatim from the original rather than regenerated, which keeps prompt / seed / steps / guidance
#   byte-identical and rules out parameter drift as an explanation for any difference.
#
# WHAT IT PROVES (and does not)
#   The injected plan is GT-future SigLIP2 -- ORACLE information, not a planner prediction. A faithful
#   reproduction therefore demonstrates the UPPER BOUND (the WM can use semantic content and the
#   conditioning path works), NOT that our planner can predict a plan this good.
#
# RESULT (2026-07-25, HPC3 job 435915, 1x GPU)
#   35 denoise steps in 4m10s -> yc74616_s0_oracle.mp4. Arm trajectory and semantic behaviour match
#   the original; not bit-identical frame by frame, which is ordinary diffusion numerical jitter
#   across hardware/driver versions. Key-frame comparison: oracle_repro/compare_gt_orig_repro.png
#
# USAGE
#   HPC3:   sbatch reproduce_oracle_yc.sh
#   direct: bash reproduce_oracle_yc.sh
#   local box: set VLM4WAM_ROOT/COSMOS_ROOT/COSMOS_VENV/WEIGHTS_DIR + the PLAN SPEC vars below,
#              and see the SEMANTIC_PLAN_* warning.
set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
COSMOS_ROOT=${COSMOS_ROOT:-$VLM4WAM_ROOT/third_party/cosmos-predict2.5}
COSMOS_VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
WEIGHTS_DIR=${WEIGHTS_DIR:-/data/user/jhe724/workspace/weights}
ORIG=${ORIG:-$VLM4WAM_ROOT/eval_results_oracle_iter3000_yellowcarrot_20260703_170049}
CKPT_PT=${CKPT_PT:-$VLM4WAM_ROOT/outputs/cosmos_semantic_plan/converted_iter3000/model_ema_bf16.pt}
OUT=${OUT:-$VLM4WAM_ROOT/eval_results_oracle_iter3000_yellowcarrot_REPRO}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_semantic_plan_320x576_93f}

cd "$COSMOS_ROOT"
mkdir -p "$VLM4WAM_ROOT/logs" "$OUT"

module load gcc/11.5 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
PY="$COSMOS_VENV/bin/python"
export VIRTUAL_ENV="$COSMOS_VENV"
export PATH="$COSMOS_VENV/bin:$PATH"
# the venv ships its own CUDA/cuDNN; loading the system cuda module instead triggers
# SUBLIBRARY_LOADING_FAILED, so point LD_LIBRARY_PATH at the bundled ones
NV="$COSMOS_VENV/lib/python3.10/site-packages/nvidia"
[ -d "$NV" ] && export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/cublas/lib:$NV/cusparse/lib:$NV/cusolver/lib:$NV/cufft/lib:$NV/curand/lib:$NV/nccl/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COSMOS_ROOT:${PYTHONPATH:-}"
export COSMOS_HF_LOCAL_DIRS="$WEIGHTS_DIR"                       # resolve HF ids to the local mirror
export COSMOS_LOCAL_MODEL_DIR="$WEIGHTS_DIR/Cosmos-Predict2.5-2B"
export SEMANTIC_PLAN_ONLINE_ENCODER_PATH=${SEMANTIC_PLAN_ONLINE_ENCODER_PATH:-$VLM4WAM_ROOT/third_party/siglip2-so400m-patch14-384}

# PLAN SPEC -- iter3000 was trained with 5 keyframes on the native 27x27 grid (3645 plan tokens).
# The HPC3 checkout defaults to exactly that, but other checkouts of cosmos-predict2.5 default to
# 6 keyframes / grid 9 (486 tokens). Those defaults silently RESAMPLE the plan into a layout the
# checkpoint has never seen, which collapses the plan cross-attention to uniform and produces a
# convincing but bogus result. Setting them explicitly costs nothing and removes the trap.
# Clip length comes from the model config, NOT the spec: generate_vid2world overrides
# num_video_frames with tokenizer.get_pixel_num_frames(config.state_t). The original run set
# COSMOS_NUM_FRAMES=49 -> state_t=13 -> 49 frames; the repo default of 93 -> state_t=24 makes
# the model extrapolate to double length, far outside what iter3000 was trained on.
export COSMOS_NUM_FRAMES=${COSMOS_NUM_FRAMES:-49}
export SEMANTIC_PLAN_NUM_KEYFRAMES=${SEMANTIC_PLAN_NUM_KEYFRAMES:-5}
export SEMANTIC_PLAN_SPATIAL_GRID=${SEMANTIC_PLAN_SPATIAL_GRID:-0}   # 0 = native 27x27

for f in "$CKPT_PT" "$ORIG/yc74616_s0.json" "$ORIG/plans/yc74616_s0_oracle.pt" \
         "$ORIG/first_frames/yc74616_f0.png" "$PY"; do
  [ -e "$f" ] || { echo "ERROR: missing required artifact: $f" >&2; exit 2; }
done

SPEC="$OUT/yc74616_s0.json"
cp "$ORIG/yc74616_s0.json" "$SPEC"      # verbatim: same prompt / seed 0 / 35 steps / guidance 7 / 49 frames
echo "=== spec ==="; cat "$SPEC"

echo "=== inference (reusing ckpt + first frame + ORACLE GT-SigLIP plan) ==="
"$PY" examples/inference.py \
  -i "$SPEC" \
  --experiment "$EXPERIMENT" \
  --checkpoint-path "$CKPT_PT" \
  --config-file cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --output-dir "$OUT" \
  --disable-guardrails 2>&1 | tee "$OUT/inference.log"

# sanity: the semantic-plan weights must actually load, and the emitted config must show 5 / 0
echo "=== semantic_plan key check (missing/unexpected should be empty) ==="
grep -iE "missing_keys|unexpected_keys|semantic_plan" "$OUT/inference.log" | head -20 || true
echo "=== plan spec actually used ==="
grep -iE "semantic_plan_num_keyframes|semantic_plan_spatial_grid" "$OUT/config.yaml" 2>/dev/null | head -3 || true

echo "=== DONE -> $OUT ==="
ls -lh "$OUT"/*.mp4 2>/dev/null
