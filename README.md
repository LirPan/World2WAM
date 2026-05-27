# World2WAM Minimal — Imagination-Distilled Fast-WAM

External wrapper project for **FastWAM + LIBERO (LeRobot)** without modifying upstream repositories.

## What this is

- **Training**: `L_total = L_action + λ_fwd · L_future` (future distillation mode)
- **Inference**: **action-only** via `FastWAM.infer_action` — no `FutureLatentHead`, no future video
- **Data**: Real FastWAM `RobotVideoDataset` (LIBERO LeRobot), not toy/random data

## Why we do not edit upstream repos

All logic lives under `idea2_workspace/minimal_world2wam/`. FastWAM / LIBERO / World2VLM / giga-world-policy are **read-only** references, imported via `sys.path`.

## Dependencies

```bash
cd idea2_workspace/minimal_world2wam
bash scripts/setup_conda_env.sh   # creates conda env `world2wam`
conda activate world2wam
```

Full experiment workflow: **[docs/RUN_EXPERIMENTS.md](docs/RUN_EXPERIMENTS.md)**

Quick framework test:

```bash
conda activate world2wam
bash scripts/smoke_test_framework.sh
```

**Long jobs (survive closing Cursor)** — run on server in background:

```bash
cd minimal_world2wam
bash scripts/bg_launch.sh full_pipeline   # download + 02/03/04
bash scripts/bg_launch.sh status          # check progress
bash scripts/bg_launch.sh tail full_pipeline
```

Logs: `experiments/bg_jobs/full_pipeline.log`

Paths in configs (relative to this directory):

- `fastwam_root: ../code/FastWAM`
- `libero_root: ../code/LIBERO`

## Modules

| Module | Role |
|--------|------|
| `src/wrappers/fastwam_wrapper.py` | `FastWAM` + MoT hook for hidden |
| `src/models/future_latent_head.py` | MLP: hidden + action → future latent |
| `src/data/libero_dataset_adapter.py` | LeRobot dataset + `future_obs` / cache |
| `src/data/future_latent_cache.py` | Disk cache for VAE targets |
| `src/losses/world2wam_losses.py` | Action + future + total loss |
| `src/train/train_fastwam_future_distill.py` | `--mode baseline` / `future_distill` |
| `src/train/train_bidirectional_world2wam.py` | `--mode forward_only` / `bidirectional` / `cycle` |
| `src/models/inverse_action_head.py` | MLP: hidden + future_latent → action |
| `src/eval/eval_action_only_fastwam.py` | Action-only metrics |
| `src/eval/eval_bidirectional_heads.py` | Offline forward / inverse / cycle MSE |

## Run code scan

```bash
bash scripts/00_scan_fastwam.sh
# Full report: notes/code_scan_report.md
```

## Precompute future latents

Requires FastWAM Wan VAE weights loaded.

```bash
bash scripts/02_precompute_future_latents.sh
# Optional: --max-samples 100
```

Caches under `data/future_latents/` (config: `cache_dir`).

If VAE is unavailable, the script **errors** with instructions — no random placeholders.

## Train baseline

**Full FastWAM training** (recommended, unchanged upstream):

```bash
bash scripts/01_run_fastwam_baseline.sh
```

This delegates to `FastWAM/scripts/train_zero1.sh`.

**Sanity forward only** (one batch):

```bash
python src/train/train_fastwam_future_distill.py \
  --config configs/fastwam_libero_baseline.yaml \
  --mode baseline
```

## Train future distillation

1. Run precompute (above)
2. Train head:

```bash
bash scripts/03_train_future_distill.sh
```

Outputs:

- `experiments/future_latent_distill/checkpoints/future_head_final.pt`
- `experiments/future_latent_distill/logs/`

## Action-only eval

```bash
bash scripts/04_eval_action_only.sh --max-batches 5
# With checkpoint:
bash scripts/04_eval_action_only.sh --checkpoint /path/to/fastwam.ckpt
```

Writes `experiments/future_latent_distill/eval_results.json` with:

- `future_head_called: false`
- `avg_latency_ms`
- `offline_action_mse` (when available)

LIBERO sim success rate: use FastWAM `experiments/libero/eval_libero_single.py` with the same checkpoint (documented in eval JSON when `--run-libero-sim`).

## Config highlights

`configs/fastwam_future_distill.yaml`:

- `hidden_dim: 1024`, `action_dim: 7`, `future_latent_dim: 48`
- `lambda_fwd: 0.1`, `future_horizon: 1`
- `use_gt_action_for_future_head: true`
- `lerobot_dataset_dirs: null` → use FastWAM default paths (override if needed)

## Interfaces still to confirm

| Item | Where to look |
|------|----------------|
| FastWAM model | `code/FastWAM/src/fastwam/models/wan22/fastwam.py` |
| Hidden (train) | MoT output `action` tokens → pool in wrapper |
| Visual encoder | `model.vae.encode` / `_encode_video_latents` |
| LIBERO batch fields | `video`, `action`, `context`, `context_mask` |
| Action inference | `model.infer_action` |

See `notes/implementation_plan.md` and `notes/code_scan_report.md`.

## Bidirectional World-Action Distillation

Three auxiliary objectives on **frozen** FastWAM hidden states (train only):

| Path | Mapping |
|------|---------|
| **Forward** | `hidden + action → future_latent` (`FutureLatentHead`) |
| **Inverse** | `hidden + future_latent → action` (`InverseActionHead`) |
| **Cycle** | `hidden + action → pred_future_latent → reconstructed_action` |

**Loss (trainable heads only):**

`loss_train_backward = λ_fwd·L_fwd + λ_inv·L_inv + λ_cycle·L_cycle`

`L_action` from FastWAM is logged as `loss_action_monitor` but does not backprop into the frozen backbone.

**Inference** remains **action-only** (`infer_action`) — neither auxiliary head is called.

```bash
# Requires 02_precompute (same cache: cache_dir + project_name world2wam_minimal)
bash scripts/05_train_bidirectional_world2wam_smoke.sh   # quick sanity
bash scripts/05_train_bidirectional_world2wam.sh         # --mode cycle

bash scripts/06_eval_bidirectional_heads.sh
bash scripts/04_eval_action_only.sh   # future_head_called & inverse_head_called: false
```

Checkpoints: `experiments/bidirectional_world2wam/checkpoints/` (`future_head_final.pt`, `inverse_head_final.pt`, `bidirectional_heads_final.pt`).

Config: `configs/bidirectional_world2wam.yaml` (`lambda_fwd`, `lambda_inv`, `lambda_cycle`, `use_gt_action_for_forward_head`).

## Next steps

1. Point `lerobot_dataset_dirs` to your LeRobot LIBERO roots
2. Download Wan + ActionDiT checkpoints per FastWAM README
3. `02_precompute` → `03_train` or `05_train_bidirectional` → `04_eval` / `06_eval`
4. Compare LIBERO success vs baseline checkpoint (bidirectional heads alone will not change sim success)
