#!/usr/bin/env bash
# Stage 2 (cosmos venv): generate videos for the ablation samples.jsonl with the
# text-free multisource model, feeding precomputed target features via the new
# target_feature_path inference field.

#SBATCH --job-name=ablation-s2
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-ablation-s2-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-ablation-s2-%j.err

set -uo pipefail
REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"
mkdir -p /data/user/jhe724/workspace/VLM4WAM/logs

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"   # this repo's cosmos_predict2 wins
NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Existing dirs so the experiment config instantiates (not iterated at inference).
export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_scene_cap200_tasktarget}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}

EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_instructsam_textfree_multisource}
# Use the consolidated .pt (inference's easy_io.load needs a .pt, not the DCP dir).
CKPT=${CKPT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_instructsam_textfree_multisource_cap200_tasktarget/cosmos_predict_v2p5/video2world/2b_droid_success_v21_instructsam_textfree_multisource_cap200_tasktarget_49f_bs2accum4_14k/checkpoints/iter_000001370/model_ema_bf16.pt}
SAMPLES=${SAMPLES:?set SAMPLES=/path/to/samples.jsonl}
OUTDIR=${OUTDIR:?set OUTDIR=/path/to/videos}

nvidia-smi -L
# tyro flattens the nested `setup` struct to top-level flags (no `setup.` prefix).
torchrun --standalone --nproc_per_node=1 examples/inference.py \
  -i "$SAMPLES" \
  --output-dir "$OUTDIR" \
  --experiment "$EXPERIMENT" \
  --checkpoint-path "$CKPT" \
  --config-file cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --disable-guardrails
status=$?
echo "stage2_exit=$status"
exit "$status"
