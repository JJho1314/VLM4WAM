#!/usr/bin/env bash
set -x
E=/data/user/jhe724/.conda/envs
rm -rf "$E/qwen35"
echo "=== rsync starVLA -> qwen35 (local FS copy, no conda, no network) ==="
rsync -a "$E/starVLA/" "$E/qwen35/"
P="$E/qwen35/bin"
echo "=== python present? ==="; "$P/python" --version
echo "=== install transformers 5.14.1 via python -m pip (internal mirror) ==="
"$P/python" -m pip install --no-input transformers==5.14.1
echo "=== verify ==="
"$P/python" -c "import torch,transformers;from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor;print('tf',transformers.__version__,'torch',torch.__version__,'tok',__import__('tokenizers').__version__,'QWEN35-OK')"
echo "COPYENVDONE rc=$?"
