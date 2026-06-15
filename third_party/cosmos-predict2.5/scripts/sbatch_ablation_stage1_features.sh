#!/usr/bin/env bash
# Stage 1 (InstructSAM conda env): extract fused multi-source features for the
# (image, query) pairs in $MANIFEST. Mirrors the precompute env setup.

#SBATCH --job-name=ablation-s1
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-ablation-s1-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-ablation-s1-%j.err

set -euo pipefail
REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"
mkdir -p /data/user/jhe724/workspace/VLM4WAM/logs

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

ISAM_ENV=${ISAM_ENV:-/data/user/jhe724/.conda/envs/instructsam}
export PATH=/data/apps/gcc/11.5/bin:$ISAM_ENV/bin:$PATH
unset PYTHONHOME
export HF_HUB_OFFLINE=1
export INSTRUCTSAM_SOURCE_ROOT=${INSTRUCTSAM_SOURCE_ROOT:-/data/user/jhe724/workspace/InstructSAM}
export INSTRUCTSAM_MODEL_PATH=${INSTRUCTSAM_MODEL_PATH:-/data/user/jhe724/workspace/InstructSAM/work_dirs/InstructSAM-2B}
export PYTHONPATH="$REPO_ROOT/scripts/_env_stubs:$REPO_ROOT:$INSTRUCTSAM_SOURCE_ROOT:${PYTHONPATH:-}"
export COSMOS_SKIP_CUDA_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false

MANIFEST=${MANIFEST:?set MANIFEST=/path/to/manifest.json}
# Use the SAME projection matrix as training so features match exactly.
PROJ_DIR=${PROJ_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_scene_cap200_tasktarget/target_features_multisource}

nvidia-smi -L
python scripts/extract_multisource_feature_pairs.py \
  --manifest "$MANIFEST" \
  --proj-dir "$PROJ_DIR" \
  --mask-tokens 16 --detect-tokens 16 --vtext-tokens 32 --out-dim 256
