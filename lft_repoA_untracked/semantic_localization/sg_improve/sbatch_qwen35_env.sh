#!/usr/bin/env bash
#SBATCH --job-name=q35-envsetup
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/q35-envsetup-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/q35-envsetup-%j.err
# conda --clone was OOM-killed on the login node; run it on a compute node (ample RAM). Clone starVLA
# (reuse its torch2.7/flash-attn/deepspeed) and only add transformers 5.x for Qwen3_5.
set -uo pipefail
source /share/anaconda3/etc/profile.d/conda.sh
echo "=== remove any half-built env ==="
conda env remove -n qwen35 -y 2>/dev/null || true
rm -rf /data/user/jhe724/.conda/envs/qwen35 2>/dev/null || true
echo "=== clone starVLA -> qwen35 ==="
conda create --clone starVLA -n qwen35 -y
P=/data/user/jhe724/.conda/envs/qwen35/bin
echo "=== install transformers 5.14.1 (internal mirror; tokenizers 0.22.2 already satisfies) ==="
"$P/pip" install --no-input transformers==5.14.1
echo "=== verify (on GPU node) ==="
"$P/python" -c "import torch,transformers;from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor;print('tf',transformers.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available(),'tok',__import__('tokenizers').__version__,'QWEN35-OK')"
echo "ENVDONE rc=$?"
