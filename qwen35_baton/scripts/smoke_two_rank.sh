#!/usr/bin/env bash
set -euo pipefail

GE_ACT_BIN="/data/LFT-W02_data/.conda/envs/ge-act/bin"
export PATH="${GE_ACT_BIN}:${PATH}"
torchrun() {
  "${GE_ACT_BIN}/python" -m torch.distributed.run "$@"
}
export QWEN35_BATON_SMOKE_INVOCATION_ID="invocation-$(\
  "${GE_ACT_BIN}/python" -c 'import uuid; print(uuid.uuid4().hex)'\
)"

torchrun --standalone --nproc_per_node=2 \
  -m qwen35_baton.cli.smoke_pipeline \
  --output-dir /tmp/qwen35_baton_two_rank \
  --verify-exact-resume

"${GE_ACT_BIN}/python" -m qwen35_baton.cli.smoke_pipeline \
  --validate-two-rank-result \
  "/tmp/qwen35_baton_two_rank/${QWEN35_BATON_SMOKE_INVOCATION_ID}/result.json"
