#!/usr/bin/env bash
# Loader throughput benchmark on one 4-GPU Worker: same short Stage-2 run with
# default loader settings, then with more workers and deeper prefetch.
set -uo pipefail
L=/data/users/junjie/workspace/VLM4WAM_baton/acp/stage2_additive.sh
export NGPU=4
RUN_ID=bench_loader_w4p2 MAX_TRAIN_STEPS=60 bash "$L"
RUN_ID=bench_loader_w8p4 MAX_TRAIN_STEPS=60 BATON_RESEARCH_NUM_WORKERS=8 BATON_RESEARCH_PREFETCH=4 bash "$L"
