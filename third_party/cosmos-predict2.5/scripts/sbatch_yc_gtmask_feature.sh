#!/usr/bin/env bash
# Extract the GT-mask oracle feature for the yellow-carrot scene (InstructSAM env;
# the script only needs SAM3 + transformers, no cosmos imports).

#SBATCH --job-name=yc-gtfeat
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:20:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/yc-gtfeat-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/yc-gtfeat-%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"
module load gcc/11.5 cuda/12.6 2>/dev/null || true

ISAM_ENV=${ISAM_ENV:-/data/user/jhe724/.conda/envs/instructsam}
export PATH=$ISAM_ENV/bin:$PATH
unset PYTHONHOME
export INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/InstructSAM}
export INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/InstructSAM/work_dirs/InstructSAM-2B}
export PYTHONPATH="$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

GTDS=${GTDS:-/data/user/jhe724/workspace/VLM4WAM/yc_ablation/gtds}
python scripts/precompute_gt_mask_target_features.py \
  --dataset-dir "$GTDS" \
  --mask-dir-name masks \
  --overwrite \
  --log-every 1
ls -la "$GTDS/target_features_gt_mask_spatial64/"
