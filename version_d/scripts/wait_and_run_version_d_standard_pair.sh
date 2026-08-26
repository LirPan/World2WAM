#!/usr/bin/env bash
set -euo pipefail

ROOT=/DATA/disk0/jianhua
POLICY_ROOT="$ROOT/latest/code/policy_lora"
PY="$ROOT/_shared/venvs/robotwin_w2wam/bin/python"
RUNNER="$POLICY_ROOT/src/eval/run_version_d_standard_pair.py"
TASKS="$POLICY_ROOT/eval_assets/fixed5_tasks.json"
OUT="$ROOT/latest/experiments/iclr_2027/robotwin_version_d_standard_pair_n3"
LOG="$OUT/waiter.log"
R0="$ROOT/third_party/FastWAM_official/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
B5="$ROOT/latest/checkpoints/ablation14_B5_fic_project14.pt"

mkdir -p "$OUT"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] waiter started"

while true; do
    GPU_ID="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F',' '{gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2+0)<=500 && ($3+0)<=5) {print $1; exit}}')"
    if [[ -n "$GPU_ID" ]]; then
        echo "[$(date -Is)] idle GPU detected: $GPU_ID; starting paired n=3"
        WORLD2WAM_GRAPH_LITE=0 WORLD2WAM_POLICY_ROOT="$POLICY_ROOT" \
        "$PY" -u "$RUNNER" \
            --episodes 3 \
            --gpu-id "$GPU_ID" \
            --tasks-json "$TASKS" \
            --output-root "$OUT" \
            --policy-root "$POLICY_ROOT" \
            --r0-ckpt "$R0" \
            --version-d-ckpt "$B5"
        RC=$?
        echo "[$(date -Is)] paired run finished rc=$RC"
        exit "$RC"
    fi
    echo "[$(date -Is)] no idle GPU; retrying in 60s"
    sleep 60
done
