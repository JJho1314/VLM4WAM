export COSMOS_CHECKPOINTS_DIR=/data/LFT-W02_data/junjie/weights

INPUT_DIR=/data/LFT-W02_data/junjie/data/cosmos_test
OUTPUT_DIR=/data/LFT-W02_data/junjie/cosmos-predict2.5/outputs/cosmos_test
CHECKPOINT_PATH=/data/LFT-W02_data/junjie/weights/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt

mkdir -p "${OUTPUT_DIR}"

for input_json in "${INPUT_DIR}"/*.json; do
  base_name="$(basename "${input_json}" .json)"
  CUDA_VISIBLE_DEVICES=0 uv run --extra=cu128 python /data/LFT-W02_data/junjie/cosmos-predict2.5/examples/inference.py \
    -i "${input_json}" \
    -o "${OUTPUT_DIR}/${base_name}" \
    --inference-type=image2world \
    --disable-guardrails \
    --checkpoint-path "${CHECKPOINT_PATH}"
done