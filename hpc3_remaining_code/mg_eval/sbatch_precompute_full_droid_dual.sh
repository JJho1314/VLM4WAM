#!/usr/bin/env bash
# Dual InstructSAM extraction (proj 256 + raw 2048 + first-frame mask PNG) over
# the FULL droid_success_v21 480x864 set (89,682 = all train + val splits).
# cap200's 17,517 are hardlinked in already; --skip-existing skips them.

#SBATCH --job-name=isam-full-dual
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-isam-full-dual-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-isam-full-dual-%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"
mkdir -p /data/user/jhe724/workspace/VLM4WAM/logs

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

ISAM_ENV=${ISAM_ENV:-/data/user/jhe724/.conda/envs/instructsam}
export PATH=/data/apps/gcc/11.5/bin:$ISAM_ENV/bin:$PATH
unset PYTHONHOME
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/InstructSAM}
export INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/VLM4WAM/models/instructsam_stage2_complete_merged}
export PYTHONPATH="$REPO_ROOT/scripts/_env_stubs:$REPO_ROOT:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export COSMOS_SKIP_CUDA_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

FULL_DIR=/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864

python - <<'PY'
import transformers
print("transformers:", transformers.__version__)
from instructsam.models import load_pretrained_model  # noqa: F401
PY
nvidia-smi -L

NPROC=${NPROC:-8}
torchrun --standalone --nproc_per_node="$NPROC" scripts/precompute_ft_features_dual.py \
  --dataset-dir "$FULL_DIR" \
  --source-root "$INSTRUCTSAM_SOURCE_ROOT" \
  --model-path "$INSTRUCTSAM_MODEL_PATH" \
  --skip-existing \
  --max-errors 3000 \
  --log-every 200
status=$?
echo "precompute_exit=$status"
ls "$FULL_DIR/target_features_ft" | grep -c "\.pt$" || true
exit "$status"
