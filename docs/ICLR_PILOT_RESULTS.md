# ICLR Pilot Results (LIBERO-Spatial, official protocol)

**Protocol**: FastWAM `eval_libero_single.py`, 10 tasks × 50 trials (overall) and hard subset tasks `{1,4,5,7,8}` × 50.  
**Floor (FastWAM official ckpt)**: overall **481/500 (96.2%)**, hard **234/250**.  
**Machine summary JSON**: [results/iclr_pilot_summary.json](results/iclr_pilot_summary.json)

## Main table

| Method | Hard | Overall | Notes |
|--------|-----:|--------:|-------|
| FastWAM reference | 234/250 | 481/500 (96.2%) | Official floor |
| **B1 action-balanced LoRA** | **244/250 (97.6%)** | 486/500 (97.2%) | Best hard-task lift |
| B2 + forward regularizer | 240/250 | 481/500 | Matches overall floor |
| B3 + forward + inverse | 239/250 | 483/500 | Modest overall lift |
| B4 naive F/I/C | 240/250 | 486/500 | Matches B1 overall |
| **B5 aligned F/I/C** | 242/250 | **489/500 (97.8%)** | Best overall |

B6 physics-gated residual as a *control* add-on was **stopped** (not used as the primary SR path).

## Takeaways

1. Beating ~96% requires adapting **ActionDiT** (LoRA) on the **official** eval path — not replacing it with a small FlowDiT head.
2. World objectives (F/I/C) help as regularizers; **B5 (aligned FIC)** is best overall; **B1 (balanced action LoRA)** is best on hard tasks.
3. Version C residual remains the clean \(\alpha\to 0\) interface story; custom-loop ~30% numbers are **invalid** for the paper table (eval mismatch).

## Code entry points (this branch)

| Item | Path |
|------|------|
| LoRA / ICLR package | [`policy_lora/`](../policy_lora/) |
| Train (action-only / hard) | `python -m src.train.train_lora_action_hard` (from `policy_lora/`) |
| Train (FIC hard) | `python -m src.train.train_lora_fic_hardtask` |
| Export merged ckpt | `python -m src.tools.export_libero_checkpoint` |
| Wrapper scripts | `scripts/train_policy_lora_fic.sh`, `scripts/export_lora_fic_official.sh` |
| Configs | `policy_lora/configs/iclr_b1_*.yaml` … `iclr_b5_*.yaml` |

```bash
# Example: train action-hard LoRA using vendored tree
cd policy_lora
export PYTHONPATH=$PWD
conda activate world2wam
python -u -m src.train.train_lora_action_hard \
  --config configs/world2wam_policy_lora_action_hard.yaml \
  --backbone-mode lora
```

## Teacher one-pager

See also [results/TEACHER_DISCUSSION_STATUS.md](results/TEACHER_DISCUSSION_STATUS.md).
