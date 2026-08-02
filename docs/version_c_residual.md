# Version C — Physics-Gated Residual World2WAM

## Idea

Keep Version A world losses (**Forward / Inverse / Cycle / Physics**), but change the **action interface**:

| | Version A | Version C |
|---|-----------|-----------|
| Flow / Inv / Cycle target | absolute \(a_{\text{GT}}\) | residual \(\delta = a_{\text{GT}} - a_{\text{FW}}\) |
| Forward input | absolute \(a_{\text{GT}}\) | **unchanged** (absolute) |
| LIBERO primary | onestep replace / absolute blend | \(a = a_{\text{FW}} + \alpha_{\text{eff}}\cdot\hat\delta\) |
| Physics role | condition tokens | condition tokens **+** gate \(\alpha(\text{phase},\text{conf})\) |

\(\alpha=0\) recovers frozen FastWAM (baseline floor). Physics opens the residual only when useful.

## Losses

\[
L_{\text{total}} =
L_{\text{flow}}(\delta)
+ \lambda_{\text{future}} L_{\text{future}}
+ \lambda_{\text{inverse}} L_{\text{inverse}}(\delta)
+ \lambda_{\text{cycle}} L_{\text{cycle}}(\delta)
+ \lambda_{\text{phase}} L_{\text{phase}}
+ \lambda_{\text{phy}} L_{\text{phy}}
\]

- Config: [`configs/world2wam_physics_residual_flow_dit_vc.yaml`](../configs/world2wam_physics_residual_flow_dit_vc.yaml) (`loss.residual_delta: true`).
- Hook: `resolve_action_flow_target` in [`train/training_utils.py`](../train/training_utils.py).

## Pipeline

```bash
# 1) Write fastwam_action into each cache .pt (multi-GPU)
bash minimal_world2wam/scripts/run_version_c_pipeline.sh precompute

# 2) Train
bash minimal_world2wam/scripts/run_version_c_pipeline.sh train

# 3) LIBERO α sweep (additive + physics gates)
bash minimal_world2wam/scripts/run_version_c_pipeline.sh eval
```

Or all-in-one: `bash minimal_world2wam/scripts/run_version_c_pipeline.sh all`

Eval mode: `--mode ours_residual_physics_flow_dit_vc` with `--residual_mode additive`.

## Metrics

| Table | Metric |
|-------|--------|
| World | MSE_fwd, cycle, phase_acc, \(\|\delta\|\) |
| Control | LIBERO SR: baseline / α=0 / residual+physics / onestep (ablation) |

Primary paper control metric: **physics-gated additive residual**, not onestep replace.
