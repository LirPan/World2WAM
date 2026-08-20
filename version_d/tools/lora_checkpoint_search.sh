#!/usr/bin/env bash
set -euo pipefail

ROOT=/DATA/disk0/yjh/robotwin_w2wam
FAST="$ROOT/third_party/FastWAM_official"
PY="$ROOT/env/bin/python"
POLICY="$ROOT/latest/code/policy_lora"
RUN="$ROOT/runs/robotwin_lora_fic_aligned/checkpoint_search"
CONFIG="$POLICY/configs/robotwin_lora_fic_aligned_newyjh.yaml"
STATS="$FAST/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
VALIDATION_TASKS=click_bell,grab_roller,handover_block
mkdir -p "$RUN"

wait_bundle() {
  local path="$1" expected_size="$2" expected_hash="$3"
  while true; do
    if [[ -f "$path" && "$(stat -c %s "$path")" == "$expected_size" ]]; then
      actual_hash="$(sha256sum "$path" | awk '{print $1}')"
      [[ "$actual_hash" == "$expected_hash" ]] && return 0
    fi
    sleep 30
  done
}

wait_bundle "$ROOT/checkpoints/R3_lora_fic_step2000_bundle.pt" 134485565 f00b71e600196f19bc5244a882eeb3ce1e01111484dbce88ef0385043565fa47
wait_bundle "$ROOT/checkpoints/R3_lora_fic_step2500_bundle.pt" 134485565 41649e1ea8971f70ecc0336b7f064109f6f5ec9c3840f279363ef9f8a0454643

run_candidate() {
  local step="$1" gpu="$2"
  local bundle="$ROOT/checkpoints/R3_lora_fic_step${step}_bundle.pt"
  local merged="$ROOT/checkpoints/R3_lora_fic_step${step}_merged.pt"
  local out="$RUN/step${step}_validation_n2"
  mkdir -p "$out"
  cd "$POLICY"
  CUDA_VISIBLE_DEVICES="$gpu" EXPORT_DEVICE=cuda "$PY" src/tools/export_libero_checkpoint.py \
    --bundle "$bundle" --config "$CONFIG" --output "$merged" > "$out/export.log" 2>&1
  cd "$FAST"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/deploy/tools/eval_robotwin_physical.py" \
    --fastwam-root "$FAST" --checkpoint "$merged" --dataset-stats "$STATS" \
    --gpu-id "$gpu" --label "R3_lora_fic_step${step}_validation_n2" --output "$out" \
    --episodes 2 --tasks "$VALIDATION_TASKS" --phase clean > "$out/driver.log" 2>&1
}

run_candidate 2000 6 &
pid2000=$!
run_candidate 2500 7 &
pid2500=$!
wait "$pid2000"
wait "$pid2500"

final_out="$RUN/final_validation_n2"
mkdir -p "$final_out"
cd "$FAST"
CUDA_VISIBLE_DEVICES=6 "$PY" "$ROOT/deploy/tools/eval_robotwin_physical.py" \
  --fastwam-root "$FAST" --checkpoint "$ROOT/checkpoints/R3_lora_fic_merged.pt" --dataset-stats "$STATS" \
  --gpu-id 6 --label R3_lora_fic_final_validation_n2 --output "$final_out" \
  --episodes 2 --tasks "$VALIDATION_TASKS" --phase clean > "$final_out/driver.log" 2>&1

best="$($PY - "$RUN/step2000_validation_n2/summary.json" "$RUN/step2500_validation_n2/summary.json" "$final_out/summary.json" <<'PY'
import json, sys
names = ("step2000", "step2500", "final")
scores = [json.load(open(path))["aggregate"]["clean"] for path in sys.argv[1:]]
best = max(range(len(scores)), key=lambda i: (scores[i], i))
print(names[best])
with open(sys.argv[1].rsplit("/", 2)[0] + "/selection.json", "w") as f:
    json.dump({"validation_tasks": ["click_bell", "grab_roller", "handover_block"], "episodes": 2,
               "scores": dict(zip(names, scores)), "selected": names[best]}, f, indent=2)
PY
)"

if [[ "$best" == final ]]; then
  exit 0
fi
selected="$ROOT/checkpoints/R3_lora_fic_${best}_merged.pt"
out="$RUN/${best}_fixed5_clean_n3"
mkdir -p "$out"
cd "$FAST"
CUDA_VISIBLE_DEVICES=6 "$PY" "$ROOT/deploy/tools/eval_robotwin_physical.py" \
  --fastwam-root "$FAST" --checkpoint "$selected" --dataset-stats "$STATS" \
  --gpu-id 6 --label "R3_lora_fic_${best}_fixed5_clean_n3" --output "$out" \
  --episodes 3 --max-tasks 5 --phase clean > "$out/driver.log" 2>&1
