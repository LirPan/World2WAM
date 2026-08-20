#!/usr/bin/env bash
set -euo pipefail

ROOT=/DATA/disk0/yjh/robotwin_w2wam
FAST="$ROOT/third_party/FastWAM_official"
POLICY="$ROOT/latest/code/policy_lora"
PY="$ROOT/env/bin/python"
CONFIG="$POLICY/configs/robotwin_lora_fic_aligned_newyjh.yaml"
STATS="$FAST/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
BUNDLE="$ROOT/checkpoints/R3_lora_fic_final_bundle.pt"
BASELINE="$FAST/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
RUN="$ROOT/runs/robotwin_lora_fic_aligned/lora_scale_hard_search"
VALIDATION_TASKS=move_stapler_pad,place_dual_shoes,turn_switch
HARD_TASKS=beat_block_hammer,move_stapler_pad,pick_dual_bottles,place_dual_shoes,press_stapler,put_object_cabinet,stack_blocks_three,stack_bowls_three,stamp_seal,turn_switch
mkdir -p "$RUN"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >> "$RUN/waiter.log"
}

pick_idle_gpu() {
  while true; do
    for gpu in 6 7; do
      row="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
      mem="${row%,*}"
      util="${row#*,}"
      if (( mem <= 1500 && util <= 5 )); then
        printf '%s\n' "$gpu"
        return 0
      fi
    done
    log "waiting for GPU 6/7"
    sleep 60
  done
}

summary_is_valid() {
  "$PY" - "$1" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
rows = p.get("rows", [])
raise SystemExit(0 if rows and all(r.get("returncode") == 0 and r.get("success") is not None for r in rows) else 1)
PY
}

export_scale() {
  local scale="$1" tag="$2" gpu="$3" merged="$4" out="$5"
  mkdir -p "$out"
  cd "$POLICY"
  CUDA_VISIBLE_DEVICES="$gpu" EXPORT_DEVICE=cuda "$PY" src/tools/export_libero_checkpoint.py \
    --bundle "$BUNDLE" --config "$CONFIG" --output "$merged" --lora-scale "$scale" \
    > "$out/export.log" 2>&1
}

evaluate() {
  local checkpoint="$1" gpu="$2" label="$3" out="$4" episodes="$5" tasks="$6" phase="$7"
  mkdir -p "$out"
  cd "$FAST"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/deploy/tools/eval_robotwin_physical.py" \
    --fastwam-root "$FAST" --checkpoint "$checkpoint" --dataset-stats "$STATS" \
    --gpu-id "$gpu" --label "$label" --output "$out" --episodes "$episodes" \
    --tasks "$tasks" --phase "$phase" > "$out/driver.log" 2>&1
  summary_is_valid "$out/summary.json"
}

run_scale_validation() {
  local scale="$1" tag="$2"
  local merged="$ROOT/checkpoints/R3_lora_fic_scale_${tag}_merged.pt"
  local out="$RUN/scale_${tag}_validation_n2"
  while true; do
    gpu="$(pick_idle_gpu)"
    log "scale=$scale validation starting on gpu=$gpu"
    if export_scale "$scale" "$tag" "$gpu" "$merged" "$out" && \
       evaluate "$merged" "$gpu" "R3_lora_fic_scale_${tag}_validation_n2" "$out" 2 "$VALIDATION_TASKS" clean; then
      log "scale=$scale validation complete"
      return 0
    fi
    log "scale=$scale failed; retrying after a GPU becomes idle"
    sleep 60
  done
}

for spec in 0.25:p025 0.5:p050 0.75:p075; do
  run_scale_validation "${spec%%:*}" "${spec##*:}"
done

final_out="$RUN/scale_p100_validation_n2"
while true; do
  gpu="$(pick_idle_gpu)"
  if evaluate "$ROOT/checkpoints/R3_lora_fic_merged.pt" "$gpu" R3_lora_fic_scale_p100_validation_n2 \
      "$final_out" 2 "$VALIDATION_TASKS" clean; then
    break
  fi
  sleep 60
done

"$PY" - "$RUN" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
specs = [(0.25, "p025"), (0.5, "p050"), (0.75, "p075"), (1.0, "p100")]
scores = {}
for scale, tag in specs:
    p = json.load(open(root / f"scale_{tag}_validation_n2" / "summary.json"))
    scores[tag] = p["aggregate"]["clean"]
selected_scale, selected_tag = max(specs, key=lambda item: (scores[item[1]], -item[0]))
with open(root / "selection.json", "w") as f:
    json.dump({"validation_tasks": ["move_stapler_pad", "place_dual_shoes", "turn_switch"],
               "episodes": 2, "scores": scores, "selected_scale": selected_scale,
               "selected_tag": selected_tag}, f, indent=2)
PY

selected_tag="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_tag"])' "$RUN/selection.json")"
if [[ "$selected_tag" == p100 ]]; then
  selected_checkpoint="$ROOT/checkpoints/R3_lora_fic_merged.pt"
else
  selected_checkpoint="$ROOT/checkpoints/R3_lora_fic_scale_${selected_tag}_merged.pt"
fi
log "selected $selected_tag checkpoint=$selected_checkpoint"

while true; do
  gpu="$(pick_idle_gpu)"
  if evaluate "$selected_checkpoint" "$gpu" "R3_lora_fic_${selected_tag}_hard10_n2_both" \
      "$RUN/selected_${selected_tag}_hard10_n2_both" 2 "$HARD_TASKS" both; then
    break
  fi
  sleep 60
done

while true; do
  gpu="$(pick_idle_gpu)"
  if evaluate "$BASELINE" "$gpu" R0_baseline_hard10_n2_both \
      "$RUN/baseline_hard10_n2_both" 2 "$HARD_TASKS" both; then
    break
  fi
  sleep 60
done

if [[ "$selected_tag" != p100 ]]; then
  while true; do
    gpu="$(pick_idle_gpu)"
    out="$RUN/selected_${selected_tag}_fixed5_clean_n3"
    mkdir -p "$out"
    cd "$FAST"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/deploy/tools/eval_robotwin_physical.py" \
      --fastwam-root "$FAST" --checkpoint "$selected_checkpoint" --dataset-stats "$STATS" \
      --gpu-id "$gpu" --label "R3_lora_fic_${selected_tag}_fixed5_clean_n3" --output "$out" \
      --episodes 3 --max-tasks 5 --phase clean > "$out/driver.log" 2>&1
    summary_is_valid "$out/summary.json" && break
    sleep 60
  done
fi
log "all scale-selection and hard-task evaluations complete"
