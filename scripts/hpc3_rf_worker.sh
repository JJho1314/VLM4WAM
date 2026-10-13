#!/usr/bin/env bash
set -euo pipefail
# Shared-home cache files can be temporarily invisible to concurrent Triton compilers.
export TRITON_CACHE_DIR
TRITON_CACHE_DIR=$(mktemp -d "/tmp/rf-triton-${SLURM_JOB_ID:?}-${LOCAL_RANK:?}.XXXXXX")
exec "$@"
