# Bidirectional World2WAM Extension Plan

## 1. Current forward-only flow

1. `LiberoDatasetAdapter` loads LIBERO LeRobot samples; `future_latent` from `FutureLatentCache` (`{cache_dir}/{project_name}/*.pt`).
2. `FastWAMWrapper.forward_train` (frozen) → MoT hook → `hidden [B, 1024]`.
3. `FutureLatentHead(hidden, GT action)` → `pred_future_latent`.
4. `L_fwd = MSE(pred, target)`; only `FutureLatentHead` is updated.
5. Inference: `forward_action_only` / `infer_action` — no auxiliary heads.

## 2. New files

| File | Role |
|------|------|
| `src/models/inverse_action_head.py` | hidden + future_latent → action |
| `src/train/train_bidirectional_world2wam.py` | `--mode forward_only \| bidirectional \| cycle` |
| `src/eval/eval_bidirectional_heads.py` | Offline head MSE / cosine |
| `configs/bidirectional_world2wam.yaml` | Full training config |
| `configs/bidirectional_world2wam_smoke.yaml` | 128 samples, few steps |
| `scripts/05_train_bidirectional_world2wam.sh` | Full cycle training |
| `scripts/05_train_bidirectional_world2wam_smoke.sh` | Smoke |
| `scripts/06_eval_bidirectional_heads.sh` | Offline eval |

## 3. Modified files

| File | Change |
|------|--------|
| `src/losses/world2wam_losses.py` | forward / inverse / cycle + `loss_train_backward` |
| `src/models/future_latent_head.py` | Docstring; config alias for GT action |
| `src/eval/eval_action_only_fastwam.py` | `inverse_head_called: false` |
| `src/models/__init__.py` | Export `InverseActionHead` |
| `README.md` | Bidirectional section |
| `notes/implementation_plan.md` | Phase update + ablations |

**Not modified:** `src/train/train_fastwam_future_distill.py`, upstream `code/FastWAM`, `code/LIBERO`.

## 4. Action-only inference isolation

- `eval_action_only_fastwam.py` only calls `run_action_only_batch` → `FastWAMWrapper.forward_action_only`.
- No import of `FutureLatentHead` or `InverseActionHead`.
- Eval JSON: `future_head_called: false`, `inverse_head_called: false`.

## 5. FastWAM backbone frozen

- `freeze_fastwam_backbone: true` → all `wrapper.model` params `requires_grad=False`.
- Optimizer: only `FutureLatentHead` / `InverseActionHead` parameters.
- `wrapper.forward_train` under `torch.no_grad()` in bidirectional trainer.
- Backward only on `loss_train_backward` (not `loss_action_monitor`).

## 6. Verify inverse and cycle

1. `bash scripts/05_train_bidirectional_world2wam_smoke.sh` — finite `loss_fwd`, `loss_inv`, `loss_cycle`.
2. Checkpoints: `future_head_final.pt`, `inverse_head_final.pt`, `bidirectional_heads_final.pt`.
3. `bash scripts/06_eval_bidirectional_heads.sh` — `offline_head_eval.json`.
4. `bash scripts/04_eval_action_only.sh` — both head flags false.

Cache: use `cache_dir: ./data/future_latents` + `project_name: world2wam_minimal`. Missing cache → raise (run `02_precompute` first).
