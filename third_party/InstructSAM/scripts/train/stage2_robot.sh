#!/usr/bin/env bash
# Single-machine InstructSAM-2B referring-seg fine-tune for the robot-tabletop domain.
# Box: 2x RTX A6000 (48GB). Targets the grounding failure: trains the SAM3 grounding
# decoder + projection heads, FREEZES the VLM (llm/vision/projector) to avoid forgetting.
#
# Prereqs:
#   1) Build data first (base miniconda python has decord+pycocotools):
#        /opt/miniconda3/bin/python tools/build_instructsam_sft_data.py \
#          --dataset-dir <YOUR_TRAIN_SPLIT_one_per_scene_dir> \
#          --out-dir data/robot_sft --frames-per-episode 6
#      ^ point at your TRAIN split, NOT a val/holdout dir you evaluate on.
#   2) SAM3 mask decoder present at $MASK_DECODER.
set -e
cd "$(dirname "$0")/../.."   # InstructSAM repo root

PY=${PY:-/data/LFT-W02_data/.conda/envs/instructsam/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

BASE_MODEL=${BASE_MODEL:-/data/LFT-W02_data/junjie/weights/CircleRadon/InstructSAM-2B}
MASK_DECODER=${MASK_DECODER:-/data/LFT-W02_data/junjie/InstructSAM/checkpoints/sam3}
DATA_DIR=${DATA_DIR:-$(pwd)/data/robot_sft}     # output of build_instructsam_sft_data.py

RUN_NAME=${RUN_NAME:-instructsam_robot_sft}
OUTPUT_DIR=work_dirs/$RUN_NAME
mkdir -p "$OUTPUT_DIR" logs

MODEL_ARGS=(
    --model_path "$BASE_MODEL"
    --mask_decoder_model "$MASK_DECODER"
    --gradient_checkpointing True
    --use_liger_kernel False
    --loss_sample_points True
)

DATA_ARGS=(
    --ann_path "$DATA_DIR/data_list.txt"
    --data_root "$DATA_DIR"
    --data_path_root "$DATA_DIR"
    --data_cache_dir "$DATA_DIR/cache"
    --model_max_length 16384
    --mm_max_length 8192
    --fps 2
    --max_frames 512
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 4          # 2 GPU x 1 x 4 = global batch 8
    --num_train_epochs 3
    --remove_unused_columns False
    --use_multi_objs False
    --skip_none False
)

# Targeted fine-tune: ONLY adapt grounding. Freeze VLM (lr=0) -> no forgetting.
# To also adapt the LLM, set --lora_enable True (and llm_lr stays 0; LoRA trains instead).
OPTIMIZER_ARGS=(
    --llm_lr 0
    --vision_encoder_lr 0
    --projector_lr 0
    --sam_decoder_lr 5e-6                     # <- the grounding decoder (DETR + mask head)
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
)

TRAINING_ARGS=(
    --deepspeed scripts/zero1.json
    --bf16 True
    --lora_enable False
    --tf32 True
    --fp16 False
    --dataloader_num_workers 8
    --loss_reduction_scope batch
    --average_tokens_across_devices False
    --group_by_modality_length True
)

LOG_ARGS=(
    --output_dir "$OUTPUT_DIR"
    --run_name "$RUN_NAME"
    --logging_steps 1
    --report_to "none"          # set "wandb" after: wandb login
    --save_strategy "steps"
    --save_steps 500
    --save_total_limit 2
)

set -x
"$PY" -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" \
    -m instructsam.train \
    "${MODEL_ARGS[@]}" "${DATA_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" "${LOG_ARGS[@]}" 2>&1 | tee -a "logs/${RUN_NAME}_local.log"
