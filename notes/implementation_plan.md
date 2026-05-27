# World2WAM Minimal — Implementation Plan

## Phase 0 — Done (scaffold)

- [x] Code scan report (`notes/code_scan_report.md`)
- [x] Directory tree under `minimal_world2wam/`
- [x] `FastWAMWrapper`, `FutureLatentHead`, losses, adapter, cache
- [x] Train / eval / shell scripts

## Phase 1 — Environment & weights (manual)

1. Install FastWAM dependencies per `code/FastWAM/README_zh.md`
2. Download Wan2.2-TI2V-5B + ActionDiT checkpoint paths in `configs/model/fastwam.yaml`
3. Prepare LeRobot LIBERO dirs (see `configs/data/libero_2cam.yaml`) or set `lerobot_dataset_dirs` in our YAML

## Phase 2 — Smoke tests

| Step | Command | Expected |
|------|---------|----------|
| Import | `cd minimal_world2wam && python -c "from src.wrappers import FastWAMWrapper"` | No import error |
| Dataset | Instantiate `build_fastwam_dataset(load_config(...))` | Len > 0 if data exists |
| VAE precompute | `bash scripts/02_precompute_future_latents.sh --max-samples 10` | `.pt` files under `cache_dir` |
| Distill 1 epoch | `bash scripts/03_train_future_distill.sh` | `future_head_final.pt` |
| Action-only eval | `bash scripts/04_eval_action_only.sh` | `eval_results.json`, `future_head_called: false` |

## Phase 3 — Interface confirmations

| ID | Item | Owner action |
|----|------|----------------|
| I1 | `mot` hook fires inside `training_loss` | Run baseline sanity; check `hidden.shape == [B, 1024]` |
| I2 | `vae.encode` on `[1,3,1,224,448]` | Run precompute on one sample |
| I3 | MoT DDP | Defer; single-GPU first |
| I4 | LIBERO sim success rate | `cd FastWAM && python experiments/libero/eval_libero_single.py ckpt=...` |
| I5 | Joint training backbone + head | Optional: unfreeze MoT after head converges |

## Phase 4 — Forward-only probe (done)

- [x] LIBERO spatial LeRobot + cached Wan VAE future latents
- [x] Frozen FastWAM MoT hidden → `FutureLatentHead`
- [x] `train_fastwam_future_distill.py` — backward on `future_loss` only
- [x] Action-only eval — no auxiliary heads

## Phase 5 — Bidirectional extension (current)

**Goal:** Forward + inverse + cycle consistency on frozen backbone.

| Mode | Script flag | Heads trained |
|------|-------------|---------------|
| forward_only | `--mode forward_only` | FutureLatentHead |
| bidirectional | `--mode bidirectional` | Future + Inverse |
| cycle | `--mode cycle` | Future + Inverse (full loop) |

```bash
bash scripts/05_train_bidirectional_world2wam_smoke.sh
bash scripts/06_eval_bidirectional_heads.sh
```

See `notes/bidirectional_extension_plan.md`.

### Why LIBERO success does not change yet

Auxiliary heads are **not** in `infer_action`. Gradients do not update ActionDiT / MoT. Sim success stays at the FastWAM checkpoint baseline until backbone is partially unfrozen.

### Next: minimal backbone unfreeze (recommended order)

1. **ActionDiT / action_expert** last 1–2 layers + `λ_action·L_action + λ_fwd·L_fwd`
2. Partial **MoT** action-token layers (VAE stays frozen)
3. Full E2E only after ablations stabilize

### Ablations to run

- `forward_only` vs `bidirectional` vs `cycle`
- `lambda_cycle` ∈ {0, 0.1, 0.5, 1.0}
- `future_horizon` ∈ {1, 2, 4}

## Phase 6 — Research extensions (later)

- Online `encode_future_latent` in training loop (no cache)
- Compare to `FastWAM.infer_joint` video branch (train only, not inference)

## File map (what to edit next)

| Goal | File |
|------|------|
| Fix hook / hidden | `src/wrappers/fastwam_wrapper.py` |
| Data paths | `configs/fastwam_future_distill.yaml` → `lerobot_dataset_dirs` |
| Loss weighting | `configs/*.yaml` → `lambda_fwd` |
| Precompute batching | `src/data/precompute_future_latents.py` |

## Known limitations

- `future_distill` training step backprops **only** `future_loss` into `FutureLatentHead` (FastWAM frozen by default).
- `pred_action` not returned from `training_loss`; offline MSE in eval may be limited.
- Baseline full training = subprocess to FastWAM `train_zero1.sh`, not reimplemented here.
