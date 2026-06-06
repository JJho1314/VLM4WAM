#!/usr/bin/env bash
# Zero-shot Cosmos-Predict2.5 2B pre-trained/base inference on the orange-cup video.

#SBATCH --job-name=cosmos-orange-zs
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:30:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-orange-base-zs-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-orange-base-zs-%j.err

set -euo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET_DIR=${DATASET_DIR:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/tavid_generation_runs/orange_cup_custom_dataset}
INPUT_VIDEO=${INPUT_VIDEO:-$DATASET_DIR/videos/orange_cup_dishwasher.mp4}
OUT=${OUT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/tavid_generation_runs/orange_cup_base_zeroshot_49f_35step}
MODEL=${MODEL:-2B/pre-trained}
NUM_FRAMES=${NUM_FRAMES:-49}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3}
SEED=${SEED:-20260524}
PROMPT=${PROMPT:-Pick up the orange cup and put it in the dishwasher.}

mkdir -p "$OUT"

cat > "$OUT/orange_cup_base_zeroshot.json" <<EOF
{
  "inference_type": "video2world",
  "name": "orange_cup_base_zeroshot",
  "input_path": "$INPUT_VIDEO",
  "prompt": "$PROMPT",
  "num_output_frames": $NUM_FRAMES,
  "num_steps": $NUM_STEPS,
  "guidance": $GUIDANCE,
  "seed": $SEED
}
EOF

cp "$INPUT_VIDEO" "$OUT/orange_cup_input_gt.mp4"
printf '%s\n' "$PROMPT" > "$OUT/orange_cup_base_zeroshot_prompt.txt"

nvidia-smi -L
python -c "import torch; print('cuda count:', torch.cuda.device_count())"
echo "model=$MODEL"
echo "input_video=$INPUT_VIDEO"
echo "output_dir=$OUT"
cat "$OUT/orange_cup_base_zeroshot.json"

torchrun --standalone --nproc_per_node=1 examples/inference.py \
  -i "$OUT/orange_cup_base_zeroshot.json" \
  -o "$OUT" \
  --model="$MODEL" \
  --disable-guardrails
