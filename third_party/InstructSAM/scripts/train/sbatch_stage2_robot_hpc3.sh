#!/usr/bin/env bash
# HPC3 (SLURM) 8-GPU InstructSAM-2B referring-seg fine-tune for the robot-tabletop domain.
# Mirrors the conventions of the project's cosmos sbatch (partition acd_u, gpu:8, modules).
#
# Targeted fine-tune: trains the SAM3 grounding decoder + projection heads, FREEZES the VLM.
# Submit:  ssh HPC3_jhe724 'cd /data/user/jhe724/workspace/InstructSAM && sbatch scripts/train/sbatch_stage2_robot_hpc3.sh'

#SBATCH --job-name=isam-robot-sft
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/InstructSAM/logs/slurm-isam-robot-%j.out
#SBATCH --error=/data/user/jhe724/workspace/InstructSAM/logs/slurm-isam-robot-%j.err

set -uo pipefail
REPO=/data/user/jhe724/workspace/InstructSAM
cd "$REPO"
mkdir -p logs

module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true

# === Training env: dedicated conda env `instructsam` (torch 2.7.1+cu128, deepspeed 0.18.5,
#     custom transformers-instructsam 5.0.0.dev0 with qwen3_vl/sam3, pycocotools, decord). ===
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate instructsam 2>/dev/null || true
PY=${PY:-/data/user/jhe724/.conda/envs/instructsam/bin/python}

BASE_MODEL=${BASE_MODEL:-$REPO/work_dirs/InstructSAM-2B}
MASK_DECODER=${MASK_DECODER:-$REPO/checkpoints/sam3}
DATA_DIR=${DATA_DIR:-$REPO/data/robot_sft}     # output of build_instructsam_sft_data.py (built on HPC3)

RUN_NAME=${RUN_NAME:-instructsam_robot_sft}
OUTPUT_DIR=work_dirs/$RUN_NAME
mkdir -p "$OUTPUT_DIR"

NPROC=${NPROC:-8}
EPOCHS=${EPOCHS:-3}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
# Tunables (defaults = targeted frozen-VLM config). LoRA: set LORA_ENABLE=True + LLM_LR>0
# (the optimizer only builds an LLM group when llm_lr>0; PEFT keeps the base VLM frozen).
LORA_ENABLE=${LORA_ENABLE:-False}
LLM_LR=${LLM_LR:-0}
PROJECTOR_LR=${PROJECTOR_LR:-0}
SAM_DECODER_LR=${SAM_DECODER_LR:-5e-6}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
"$PY" -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node "$NPROC" \
  -m instructsam.train \
  --model_path "$BASE_MODEL" --mask_decoder_model "$MASK_DECODER" \
  --attn_implementation sdpa \
  --gradient_checkpointing True --use_liger_kernel False --loss_sample_points True \
  --ann_path "$DATA_DIR/data_list.txt" --data_root "$DATA_DIR" --data_path_root "$DATA_DIR" --data_cache_dir "$DATA_DIR/cache" \
  --model_max_length 16384 --mm_max_length 8192 --fps 2 --max_frames 512 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --num_train_epochs "$EPOCHS" --remove_unused_columns False --use_multi_objs False --skip_none False \
  --llm_lr "$LLM_LR" --vision_encoder_lr 0 --projector_lr "$PROJECTOR_LR" --sam_decoder_lr "$SAM_DECODER_LR" \
  --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --deepspeed scripts/zero1.json --bf16 True --lora_enable "$LORA_ENABLE" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --tf32 True --fp16 False \
  --dataloader_num_workers 8 --loss_reduction_scope batch --average_tokens_across_devices False \
  --group_by_modality_length True \
  --output_dir "$OUTPUT_DIR" --run_name "$RUN_NAME" \
  --logging_steps 1 --report_to none --save_strategy "$SAVE_STRATEGY" --save_steps 500 --save_total_limit 2 \
  ${MAX_STEPS:+--max_steps "$MAX_STEPS"}
