#!/usr/bin/env bash
# Mask-free behavioral eval of the match-ground model: features only, no masks.
#SBATCH --job-name=mg-eval-t
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-mg-eval-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-mg-eval-%j.err
set -uo pipefail
REPO_ROOT=/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5
cd "$REPO_ROOT"
module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true
VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS=/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_scene_cap200_tasktarget
export DROID_SUCCESS_V21_TAVID_DIR=$DS
export DROID_SUCCESS_V21_TAVID_VAL_DIR=/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3

CKPT=/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_match_ground_2000_vlm4wam/cosmos_predict_v2p5/video2world/2b_droid_success_v21_match_ground_cap200_49f_bs2accum8_gbs128_2000/checkpoints/iter_000002000/model_ema_bf16.pt
echo "=== trained gate / matching head sanity ==="
python - <<PYEOF
import torch, math
sd = torch.load("$CKPT", map_location="cpu", weights_only=False)
sd = sd.get("model", sd) if isinstance(sd, dict) else sd
hits = {k: v for k, v in sd.items() if "match_ground" in k}
for k in sorted(hits):
    t = hits[k]
    if t.numel() == 1:
        print(f"{k}: value={float(t):.4f} tanh={math.tanh(float(t)):.4f}")
    else:
        print(f"{k}: shape={tuple(t.shape)} norm={t.float().norm():.3f}")
del sd
PYEOF
torchrun --standalone --nproc_per_node=1 examples/inference.py \
  -i /data/user/jhe724/workspace/VLM4WAM/mg_eval/samples_mg_text.jsonl \
  --output-dir /data/user/jhe724/workspace/VLM4WAM/mg_eval/videos_text \
  --experiment predict2_video2world_training_2b_droid_success_v21_match_ground \
  --checkpoint-path "$CKPT" \
  --config-file cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --disable-guardrails
echo "infer_exit=$?"
ls -la /data/user/jhe724/workspace/VLM4WAM/mg_eval/videos_text/
