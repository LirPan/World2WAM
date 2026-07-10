#!/usr/bin/env bash
# Fresh download + extract, then run debug experiment pipeline.
set -euo pipefail

WORKSPACE="/DATA/disk0/jianhua"
export FRESH_DOWNLOAD=1

echo "== redownload_and_pipeline: $(date -Iseconds) =="
cd "${WORKSPACE}"
bash minimal_world2wam/scripts/setup_deps.sh data_fresh
echo "== dataset OK, starting experiment pipeline =="
bash minimal_world2wam/scripts/wait_and_run_pipeline.sh
