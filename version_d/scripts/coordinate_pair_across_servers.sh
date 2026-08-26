#!/usr/bin/env bash
set -euo pipefail

LEASE_DIR=/Users/panlaoba/Desktop/ICLR/.world2wam_version_d_pair_lease
LOG=/Users/panlaoba/Desktop/ICLR/world2wam_version_d_pair_coordinator.log
EXPECTED_B5=12042181884

timestamp() {
    date '+%F %T%z'
}

if ! mkdir "$LEASE_DIR" 2>/dev/null; then
    echo "[$(timestamp)] coordinator already active or lease exists" >>"$LOG"
    exit 0
fi
exec >>"$LOG" 2>&1
echo "[$(timestamp)] dual-server coordinator started"

idle_gpu() {
    local host="$1"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F',' '{gsub(/ /,\"\",\$2); gsub(/ /,\"\",\$3); if ((\$2+0)<=500 && (\$3+0)<=5) {print \$1; exit}}'" \
        2>/dev/null || true
}

launch_fiveages() {
    local gpu="$1"
    ssh FiveAges_A100_2 "mkdir -p /DATA/disk0/jianhua/latest/experiments/iclr_2027/robotwin_version_d_standard_pair_n3; nohup env WORLD2WAM_GRAPH_LITE=0 /DATA/disk0/jianhua/_shared/venvs/robotwin_w2wam/bin/python -u /DATA/disk0/jianhua/latest/code/policy_lora/src/eval/run_version_d_standard_pair.py --episodes 3 --gpu-id $gpu --tasks-json /DATA/disk0/jianhua/latest/code/policy_lora/eval_assets/fixed5_tasks.json --output-root /DATA/disk0/jianhua/latest/experiments/iclr_2027/robotwin_version_d_standard_pair_n3 --policy-root /DATA/disk0/jianhua/latest/code/policy_lora --r0-ckpt /DATA/disk0/jianhua/third_party/FastWAM_official/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt --version-d-ckpt /DATA/disk0/jianhua/latest/checkpoints/ablation14_B5_fic_project14.pt > /DATA/disk0/jianhua/latest/experiments/iclr_2027/robotwin_version_d_standard_pair_n3/coordinator_run.log 2>&1 & echo launched_pid=\$!"
}

launch_new_yjh() {
    local gpu="$1"
    ssh New_yjh "mkdir -p /DATA/disk0/yjh/robotwin_w2wam/runs/version_d_standard_pair_n3; nohup env WORLD2WAM_GRAPH_LITE=0 /DATA/disk0/yjh/robotwin_w2wam/env/bin/python -u /DATA/disk0/yjh/robotwin_w2wam/run_version_d_standard_pair_new_yjh.py --episodes 3 --gpu-id $gpu --output-root /DATA/disk0/yjh/robotwin_w2wam/runs/version_d_standard_pair_n3 > /DATA/disk0/yjh/robotwin_w2wam/runs/version_d_standard_pair_n3/coordinator_run.log 2>&1 & echo launched_pid=\$!"
}

while true; do
    five_gpu="$(idle_gpu FiveAges_A100_2)"
    if [[ -n "$five_gpu" ]]; then
        echo "[$(timestamp)] FiveAges idle GPU=$five_gpu; claiming it"
        launch_fiveages
        echo "[$(timestamp)] launched on FiveAges"
        exit 0
    fi

    new_b5_size="$(ssh -o BatchMode=yes -o ConnectTimeout=10 New_yjh "stat -c '%s' /DATA/disk0/yjh/robotwin_w2wam/checkpoints/version_d_transfer/ablation14_B5_fic_project14.pt 2>/dev/null || true" 2>/dev/null || true)"
    new_gpu="$(idle_gpu New_yjh)"
    if [[ "$new_b5_size" == "$EXPECTED_B5" && -n "$new_gpu" ]]; then
        echo "[$(timestamp)] New_yjh idle GPU=$new_gpu and B5 transfer complete; claiming it"
        launch_new_yjh
        echo "[$(timestamp)] launched on New_yjh"
        exit 0
    fi
    echo "[$(timestamp)] no eligible idle GPU; FiveAges_gpu=${five_gpu:-none}, New_yjh_gpu=${new_gpu:-none}, New_yjh_B5_size=${new_b5_size:-missing}"
    sleep 60
done
