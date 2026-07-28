#!/usr/bin/env bash
#SBATCH --job-name=sig2_pca_probe
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jhe724/junjie/logs/sig2-pca-probe-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af}"
PYTHON="${PYTHON:-/data/user/jhe724/.venvs/vlm4wam_joint/bin/python}"
FRAME_CACHE_DIR="${FRAME_CACHE_DIR:-/data/user/jhe724/workspace/data/libero_fastwam_frame_cache_160}"
SIGLIP2_MODEL_DIR="${SIGLIP2_MODEL_DIR:-/data/user/jhe724/junjie/weights/siglip2-large-patch16-256}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/user/jhe724/junjie/probes_2b/siglip2_pca_upsample}"
RUN_KIND="${RUN_KIND:-formal}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ "$RUN_KIND" == "smoke" ]]; then
    STEPS="${STEPS:-2}"
    PCA_BATCHES="${PCA_BATCHES:-1}"
    VALIDATION_BATCHES="${VALIDATION_BATCHES:-1}"
else
    STEPS="${STEPS:-5000}"
    PCA_BATCHES="${PCA_BATCHES:-25}"
    VALIDATION_BATCHES="${VALIDATION_BATCHES:-50}"
fi

for path in \
    "$PYTHON" \
    "$FRAME_CACHE_DIR" \
    "$SIGLIP2_MODEL_DIR/config.json"; do
    [[ -e "$path" ]] || {
        echo "missing required path: $path" >&2
        exit 2
    }
done

mkdir -p "$OUTPUT_DIR" /data/user/jhe724/junjie/logs
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12

exec "$PYTHON" \
    "$REPO_ROOT/qwen3_vl_semantic_planner/dinov3_da3_2b/train_siglip2_pca_probe.py" \
    --frame-cache-dir "$FRAME_CACHE_DIR" \
    --siglip2-model-dir "$SIGLIP2_MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --pca-batches "$PCA_BATCHES" \
    --validation-batches "$VALIDATION_BATCHES" \
    --device cuda
