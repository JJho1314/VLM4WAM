#!/usr/bin/env bash
set -x
source /share/anaconda3/etc/profile.d/conda.sh
echo "=== CLONE START ==="
conda create --clone starVLA -n qwen35 -y
P=/data/user/jhe724/.conda/envs/qwen35/bin
echo "=== INSTALL transformers 5.14.1 ==="
"$P/pip" install --no-input transformers==5.14.1
echo "=== VERIFY ==="
"$P/python" -c "import torch,transformers;from transformers import Qwen3_5ForConditionalGeneration;print('tf',transformers.__version__,'torch',torch.__version__,'tok',__import__('tokenizers').__version__,'QWEN35-OK')"
echo "ENVDONE rc=$?"
