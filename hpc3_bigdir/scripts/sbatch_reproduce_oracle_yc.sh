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
# Reproduce eval_results_oracle_iter3000_yellowcarrot_20260703_170049 by re-running ONLY inference,
# reusing the existing converted checkpoint + first frame + oracle GT-SigLIP plan (the two missing
# helper scripts only PRODUCE those, which already exist). seed=0 -> should match the original.
set -euo pipefail
VLM4WAM_ROOT=/data/user/jhe724/workspace/VLM4WAM
COSMOS_ROOT=$VLM4WAM_ROOT/third_party/cosmos-predict2.5
cd "$COSMOS_ROOT"; mkdir -p "$VLM4WAM_ROOT/logs"
module load gcc/11.5 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
COSMOS_VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
PY="$COSMOS_VENV/bin/python"
export VIRTUAL_ENV="$COSMOS_VENV"; export PATH="$COSMOS_VENV/bin:$PATH"
NV="$COSMOS_VENV/lib/python3.10/site-packages/nvidia"
[ -d "$NV" ] && export LD_LIBRARY_PATH="$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/cublas/lib:$NV/cusparse/lib:$NV/cusolver/lib:$NV/cufft/lib:$NV/curand/lib:$NV/nccl/lib:$NV/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$COSMOS_ROOT:${PYTHONPATH:-}"
export COSMOS_HF_LOCAL_DIRS=/data/user/jhe724/workspace/weights
export COSMOS_LOCAL_MODEL_DIR=/data/user/jhe724/workspace/weights/Cosmos-Predict2.5-2B
export SEMANTIC_PLAN_ONLINE_ENCODER_PATH=$VLM4WAM_ROOT/third_party/siglip2-so400m-patch14-384

ORIG=$VLM4WAM_ROOT/eval_results_oracle_iter3000_yellowcarrot_20260703_170049
CKPT_PT=$VLM4WAM_ROOT/outputs/cosmos_semantic_plan/converted_iter3000/model_ema_bf16.pt
OUT=$VLM4WAM_ROOT/eval_results_oracle_iter3000_yellowcarrot_REPRO
mkdir -p "$OUT"
SPEC=$OUT/yc74616_s0.json
cp "$ORIG/yc74616_s0.json" "$SPEC"

echo "=== inference (reuse ckpt + first-frame + oracle GT plan) ==="
"$PY" examples/inference.py \
  -i "$SPEC" \
  --experiment predict2_video2world_training_2b_droid_semantic_plan_320x576_93f \
  --checkpoint-path "$CKPT_PT" \
  --config-file cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --output-dir "$OUT" \
  --disable-guardrails 2>&1 | tee "$OUT/inference.log"
echo "=== DONE ==="
ls -lh "$OUT"/*.mp4 2>/dev/null
