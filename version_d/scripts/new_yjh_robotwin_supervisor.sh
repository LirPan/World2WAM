#!/usr/bin/env bash
# Persistent, non-invasive RoboTwin R0-R3 experiment supervisor for New_yjh.
set -uo pipefail

ROOT=/DATA/disk0/yjh/robotwin_w2wam
ENV_PREFIX="$ROOT/env"
PY="$ENV_PREFIX/bin/python"
CONDA=/home/yjh/miniconda3/bin/conda
FAST="$ROOT/third_party/FastWAM_official"
POLICY="$ROOT/latest/code/policy_lora"
DATA="$ROOT/data/robotwin2.0"
RUN="$ROOT/runs/robotwin_lora_fic_aligned"
RELEASE="$FAST/checkpoints/fastwam_release"
LOG="$RUN/logs/supervisor.log"
CANDIDATES=(0 1 4 5)

mkdir -p "$RUN"/{logs,status,audits,eval,R0,R1,R2,R3} "$ROOT"/{checkpoints,data,cache}
exec >>"$LOG" 2>&1
echo "[$(date -Is)] supervisor started"

mark() { echo "[$(date -Is)] $*"; }
retry() {
  local attempts="$1"; shift
  local count=1
  until "$@"; do
    if (( attempts > 0 && count >= attempts )); then
      mark "FAILED after ${count} attempts: $*"
      return 1
    fi
    mark "retry ${count}/${attempts} failed: $*; sleeping 60s"
    count=$((count + 1))
    sleep 60
  done
}
gpu_idle() {
  local gpu="$1" memory util
  IFS=',' read -r memory util < <(nvidia-smi --id="$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  [[ "$memory" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$memory" -le 500 && "$util" -le 5 ]]
}
wait_for_gpu() {
  local gpu="$1"
  while ! gpu_idle "$gpu"; do
    mark "GPU ${gpu} is not idle; will not disturb its owner. Retrying in 60s."
    sleep 60
  done
}
wait_for_all() {
  local gpu
  while true; do
    for gpu in "${CANDIDATES[@]}"; do gpu_idle "$gpu" || { mark "waiting for preferred GPUs: ${CANDIDATES[*]}"; sleep 60; continue 2; }; done
    return 0
  done
}
state() { printf '%s\n' "$2" > "$RUN/status/$1"; }

bootstrap_env() {
  if [[ ! -x "$PY" ]]; then
    mark "creating isolated conda environment"
    retry 0 "$CONDA" create -p "$ENV_PREFIX" python=3.10 -y || return 1
  fi
  if ! "$PY" -c 'import fastwam, peft, yaml, huggingface_hub' >/dev/null 2>&1; then
    retry 0 "$PY" -m pip install --upgrade pip || return 1
    retry 0 "$PY" -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e "$FAST" --no-build-isolation || return 1
    retry 0 "$PY" -m pip install peft pyyaml huggingface_hub || return 1
  else
    mark "isolated Python environment already has FastWAM and download dependencies"
  fi
  if ! "$PY" -c 'import sapien, mplib, curobo' >/dev/null 2>&1; then
    mark "installing official RoboTwin simulator runtime"
    retry 0 env PATH="$ENV_PREFIX/bin:$PATH" bash "$FAST/third_party/RoboTwin/script/_install.sh" || return 1
  fi
  "$PY" -c 'import torch, peft, sapien, mplib, curobo; print(torch.__version__, torch.cuda.device_count())' || return 1
}

download_assets() {
  # The server's SOCKS5 proxy is reachable for the mirror API and avoids
  # repeated direct-connection timeouts during large Hub downloads.
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_DISABLE_TELEMETRY=1 DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
  export ROBOTWIN_SOCKS_PROXY="${ROBOTWIN_SOCKS_PROXY:-socks5h://127.0.0.1:1080}"
  export HTTP_PROXY="$ROBOTWIN_SOCKS_PROXY" HTTPS_PROXY="$ROBOTWIN_SOCKS_PROXY" ALL_PROXY="$ROBOTWIN_SOCKS_PROXY"
  mkdir -p "$RELEASE" "$DATA"
  if [[ ! -f "$RELEASE/robotwin_uncond_3cam_384.pt" || ! -f "$RELEASE/robotwin_uncond_3cam_384_dataset_stats.json" ]]; then
    retry 0 env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY "$PY" -m pip install pysocks || return 1
    retry 0 "$PY" -m huggingface_hub.commands.huggingface_cli download yuanty/fastwam robotwin_uncond_3cam_384.pt robotwin_uncond_3cam_384_dataset_stats.json --local-dir "$RELEASE" || return 1
  else
    mark "FastWAM checkpoint and dataset stats already complete"
  fi
  if ! "$PY" -c 'import socks' >/dev/null 2>&1; then
    retry 0 env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY "$PY" -m pip install pysocks || return 1
  fi
  if [[ ! -f "$DATA/robotwin2.0/meta/tasks.jsonl" ]]; then
    if [[ "${FAST_FIRST:-1}" == "1" ]]; then
      if [[ ! -f "$RUN/status/robotwin_data_download.pid" ]] || ! kill -0 "$(cat "$RUN/status/robotwin_data_download.pid")" 2>/dev/null; then
        mark "starting full RoboTwin data download in background; fast smoke path continues"
        download_robotwin_dataset >"$RUN/logs/robotwin_data_download.log" 2>&1 &
        echo "$!" > "$RUN/status/robotwin_data_download.pid"
      fi
      return 0
    fi
    download_robotwin_dataset || return 1
  fi
  [[ -f "$DATA/robotwin2.0/meta/tasks.jsonl" ]] || { mark "RoboTwin tasks.jsonl is missing after download"; return 1; }
  ln -sfn "$RELEASE/robotwin_uncond_3cam_384_dataset_stats.json" "$DATA/dataset_stats.json"
  mkdir -p "$FAST/data"
  ln -sfn "$DATA" "$FAST/data/robotwin2.0"
}

download_robotwin_dataset() {
  download_robotwin_parts || return 1
  if [[ ! -f "$DATA/robotwin2.0/meta/tasks.jsonl" ]] && compgen -G "$DATA/robotwin2.0.tar.gz.part-*" >/dev/null; then
    mark "extracting RoboTwin data archive"
    (cd "$DATA" && cat robotwin2.0.tar.gz.part-* | tar -xzf -) || return 1
  fi
  [[ -f "$DATA/robotwin2.0/meta/tasks.jsonl" ]] || return 1
  ln -sfn "$RELEASE/robotwin_uncond_3cam_384_dataset_stats.json" "$DATA/dataset_stats.json"
  mkdir -p "$FAST/data"
  ln -sfn "$DATA" "$FAST/data/robotwin2.0"
}

download_robotwin_parts() {
  local file url pid status=0
  local files=(
    robotwin2.0.tar.gz.part-00 robotwin2.0.tar.gz.part-01
    robotwin2.0.tar.gz.part-02 robotwin2.0.tar.gz.part-03
    robotwin2.0.tar.gz.part-04 robotwin2.0.tar.gz.part-05
    robotwin2.0.tar.gz.part-06 robotwin2.0.tar.gz.part-07
  )
  mkdir -p "$DATA/download_logs"
  mark "downloading RoboTwin's 8 archive parts through the verified SOCKS5 proxy"
  for file in "${files[@]}"; do
    if [[ -f "$DATA/$file" && "$(stat -c %s "$DATA/$file" 2>/dev/null || echo 0)" -gt 1000000000 ]]; then
      mark "resume check: $file already present"
      continue
    fi
    url="https://hf-mirror.com/datasets/yuanty/robotwin2.0-fastwam/resolve/main/$file"
    (
      curl --proxy "$HTTPS_PROXY" --fail --location --retry 30 --retry-all-errors --retry-delay 10 --continue-at - --output "$DATA/$file" "$url"
    ) >"$DATA/download_logs/$file.log" 2>&1 &
    echo "$!" > "$DATA/download_logs/$file.pid"
  done
  for file in "${files[@]}"; do
    [[ -f "$DATA/download_logs/$file.pid" ]] || continue
    pid=$(cat "$DATA/download_logs/$file.pid")
    wait "$pid" || status=1
  done
  return "$status"
}

prepare_assets() {
  local action_dit="$FAST/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
  if [[ ! -f "$action_dit" ]]; then
    wait_for_gpu 0
    mark "preparing ActionDiT assets using GPU 0"
    retry 0 env DIFFSYNTH_MODEL_BASE_PATH="$FAST/checkpoints" CUDA_VISIBLE_DEVICES=0 "$PY" "$FAST/scripts/preprocess_action_dit_backbone.py" --model-config "$FAST/configs/model/fastwam.yaml" --output "$action_dit" --device cuda --dtype bfloat16 || return 1
  fi
  retry 0 "$PY" "$FAST/scripts/precompute_text_embeds.py" task=robotwin_uncond_3cam_384_1e-4 overwrite=false || return 1
}

precompute_cache() {
  local gpu shard pid status=0
  wait_for_all
  mark "launching four future-latent cache shards"
  for shard in 0 1 2 3; do
    gpu="${CANDIDATES[$shard]}"
    (
      cd "$POLICY" || exit 1
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" src/data/precompute_future_latents.py --config configs/robotwin_r3_lora_fic_projection.yaml --max-samples 4096 --device cuda --shard-id "$shard" --num-shards 4
    ) >"$RUN/logs/cache_shard_${shard}.log" 2>&1 &
    printf '%s\n' "$!" > "$RUN/status/cache_shard_${shard}.pid"
  done
  for shard in 0 1 2 3; do
    pid=$(cat "$RUN/status/cache_shard_${shard}.pid")
    wait "$pid" || status=1
  done
  return "$status"
}

evaluate() {
  local label="$1" gpu="$2" checkpoint="$3"
  wait_for_gpu "$gpu"
  "$PY" "$ROOT/deploy/tools/eval_robotwin_physical.py" --fastwam-root "$FAST" --checkpoint "$checkpoint" --dataset-stats "$RELEASE/robotwin_uncond_3cam_384_dataset_stats.json" --gpu-id "$gpu" --label "$label" --output "$RUN/eval/$label"
}
train_export_eval() {
  local label="$1" gpu="$2" config="$3" trainer="$4" gradient="$5"
  wait_for_gpu "$gpu"
  cd "$POLICY" || return 1
  local command=("$PY" "$trainer" --config "$config" --backbone-mode lora --max-steps 3000)
  [[ -n "$gradient" ]] && command+=(--gradient-mode "$gradient")
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" || return 1
  CUDA_VISIBLE_DEVICES="$gpu" EXPORT_DEVICE=cuda "$PY" src/tools/export_libero_checkpoint.py --bundle "$RUN/$label/checkpoints/world2wam_final.pt" --config "$config" --output "$ROOT/checkpoints/$label.pt" || return 1
  evaluate "$label" "$gpu" "$ROOT/checkpoints/$label.pt"
}

run_method() {
  local label="$1" gpu="$2" mode="$3"
  state "$label.state" "running"
  if [[ "$mode" == "baseline" ]]; then
    evaluate "$label" "$gpu" "$RELEASE/robotwin_uncond_3cam_384.pt"
  elif [[ "$label" == "R1" ]]; then
    train_export_eval R1 "$gpu" configs/robotwin_r1_lora.yaml src/train/train_lora_action_hard.py ""
  elif [[ "$label" == "R2" ]]; then
    train_export_eval R2 "$gpu" configs/robotwin_r2_lora_fic.yaml src/train/train_lora_fic_hardtask.py naive
  else
    train_export_eval R3 "$gpu" configs/robotwin_r3_lora_fic_projection.yaml src/train/train_lora_fic_hardtask.py project_conflicts
  fi
  local result=$?
  if [[ "$result" -eq 0 ]]; then state "$label.state" "complete"; else state "$label.state" "failed"; fi
  return "$result"
}

main() {
  state supervisor.state "bootstrapping"
  bootstrap_env || { state supervisor.state "failed_bootstrap"; return 1; }
  state supervisor.state "downloading"
  download_assets || { state supervisor.state "failed_download"; return 1; }
  state supervisor.state "preparing_assets"
  prepare_assets || { state supervisor.state "failed_asset_prep"; return 1; }
  if [[ ! -f "$DATA/robotwin2.0/meta/tasks.jsonl" ]]; then
    state supervisor.state "fast_r0_smoke"
    run_method R0 0 baseline >"$RUN/logs/R0.log" 2>&1 || true
    state supervisor.state "waiting_for_full_dataset"
    while [[ ! -f "$DATA/robotwin2.0/meta/tasks.jsonl" ]]; do
      data_pid=$(cat "$RUN/status/robotwin_data_download.pid" 2>/dev/null || true)
      if [[ -n "$data_pid" ]] && ! kill -0 "$data_pid" 2>/dev/null; then
        mark "background dataset worker exited before tasks.jsonl; restarting it"
        download_robotwin_dataset >"$RUN/logs/robotwin_data_download.log" 2>&1 &
        echo "$!" > "$RUN/status/robotwin_data_download.pid"
      fi
      sleep 60
    done
  fi
  "$PY" "$ROOT/deploy/tools/audit_robotwin_tasks.py" --tasks "$DATA/robotwin2.0/meta/tasks.jsonl" --output "$RUN/audits/hard_tasks.json" || return 1
  state supervisor.state "precomputing"
  precompute_cache || { state supervisor.state "failed_cache"; return 1; }
  state supervisor.state "running_r0_r3"
  wait_for_all
  run_method R0 0 baseline >"$RUN/logs/R0.log" 2>&1 & pid0=$!
  run_method R1 1 train >"$RUN/logs/R1.log" 2>&1 & pid1=$!
  run_method R2 4 train >"$RUN/logs/R2.log" 2>&1 & pid2=$!
  run_method R3 5 train >"$RUN/logs/R3.log" 2>&1 & pid3=$!
  printf '%s\n' "$pid0 $pid1 $pid2 $pid3" > "$RUN/status/method_pids"
  local status=0 pid
  for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do wait "$pid" || status=1; done
  "$PY" "$ROOT/deploy/tools/summarize_matrix.py" --root "$RUN" || status=1
  if [[ "$status" -eq 0 ]]; then state supervisor.state "complete"; else state supervisor.state "completed_with_failures"; fi
  return "$status"
}

main "$@"
