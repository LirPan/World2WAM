# policy_lora — ActionDiT LoRA + world regularizers

Vendored from the idea2 `minimal_world2wam/src` tree for the ICLR control path.

## Layout

- `src/train/` — `train_lora_action_hard.py`, `train_lora_fic_hardtask.py`, …
- `src/wrappers/backbone_modes.py` — PEFT LoRA on action expert
- `src/tools/export_libero_checkpoint.py` — merge LoRA → official FastWAM `.pt`
- `configs/` — ICLR B1–B5 + hard-task LoRA YAMLs

## Run

```bash
cd policy_lora
export PYTHONPATH=$PWD
conda activate world2wam
python -u -m src.train.train_lora_action_hard \
  --config configs/iclr_b1_action_balanced.yaml \
  --backbone-mode lora
```

Or from repo root:

```bash
bash scripts/train_policy_lora_fic.sh --max-steps 200
```

External deps (not in this folder): FastWAM checkpoint, LIBERO, LeRobot data paths in the YAML.
