#!/usr/bin/env bash
# Local (non-SLURM) launcher for Stage 1 on a single machine.
# This box: 2x RTX A6000 (48GB). Adjust NPROC_PER_NODE / CUDA_VISIBLE_DEVICES as needed.
#
# Prerequisites still required before this can run end-to-end (NOT downloaded yet):
#   - Base model:   checkpoints/Qwen3-VL-2B-Instruct   (HF: Qwen/Qwen3-VL-2B-Instruct)
#   - Mask decoder: checkpoints/sam3
#   - Training data + annotation JSONs under data/training/ (see data/stage1.txt)
set -e
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

WORK_DIR=work_dirs
RUN_NAME=instructsam_stage1_2b
OUTPUT_DIR=$WORK_DIR/$RUN_NAME
mkdir -p "$OUTPUT_DIR" logs

MODEL_ARGS=(
    --model_path checkpoints/Qwen3-VL-2B-Instruct
    --mask_decoder_model checkpoints/sam3
    --gradient_checkpointing True
    --use_liger_kernel False
    --loss_sample_points True
)

DATA_ARGS=(
    --ann_path ./data/stage1.txt
    --data_root ./data
    --data_path_root ./data/training/
    --data_cache_dir ./data/cache
    --model_max_length 16384
    --mm_max_length 8192
    --fps 2
    --max_frames 512
    --per_device_train_batch_size 1
    # Original 8-GPU run used global batch = 8 * 1 * 2 = 16.
    # On 2 GPUs, raise gradient_accumulation_steps to 8 to match (2 * 1 * 8 = 16).
    --gradient_accumulation_steps 8
    --num_train_epochs 1
    --remove_unused_columns False
    --use_multi_objs True
    --skip_none False
)

OPTIMIZER_ARGS=(
    --llm_lr 5e-6
    --projector_lr 5e-6
    --vision_encoder_lr 5e-6
    --sam_decoder_lr 1e-5
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
)

TRAINING_ARGS=(
    --deepspeed scripts/zero1.json
    --bf16 True
    --lora_enable True
    --tf32 True
    --fp16 False
    --dataloader_num_workers 8
    --loss_reduction_scope batch
    --average_tokens_across_devices False
    --group_by_modality_length True
)

LOG_ARGS=(
    --output_dir $OUTPUT_DIR
    --run_name $RUN_NAME
    --logging_steps 1
    --report_to "none"   # set to "wandb" once logged in: wandb login
    --save_strategy "steps"
    --save_steps 1000
    --save_total_limit 2
)

set -x
torchrun --standalone --nnodes 1 --nproc_per_node $NPROC_PER_NODE \
    -m instructsam.train \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${LOG_ARGS[@]}" 2>&1 | tee -a logs/${RUN_NAME}_local.log
