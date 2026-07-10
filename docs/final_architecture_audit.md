# Final Architecture Audit (Version A)

**Date:** 2026-07-09  
**Project:** Physics-Aligned World2WAM  
**Goal:** Decouple world understanding from action generation; physics-aligned latent imagination + flow-based multimodal actions.

---

## Target Architecture

```
Instruction → Text Encoder
Current z_t ──┬── StudentPhysicsRouter(z_t, text, state_t) → physics_code
              ├── FutureWorldHead(z_t, action, text, physics) → z_hat_future
              └── FlowActionDiT(z_t, text, physics) → action_chunk   [NO z_tH]

TeacherPhysicsLabeler(z_t, z_tH, action, state) → phase_id + confidence  [train only]

Inverse / Cycle: same FlowActionDiT with future_latent = z_tH or z_pred_H
```

**Loss:**

```
L_total = L_flow + λ_future·L_future + λ_inverse·L_inverse + λ_cycle·L_cycle + λ_phase·L_phase + λ_phy·L_phy
```

---

## Pre-Refactor Gap Analysis

| Component | Pre-refactor | Target | Status |
|-----------|--------------|--------|--------|
| Future World Head | ForwardHead MLP + MSE | Same | OK |
| Main action decoder | MLP default; FlowDiT opt-in | FlowActionDiT default | **Fixed** |
| Inverse | InverseHead MLP + MSE | Shared FlowDiT + flow loss | **Fixed** |
| Cycle | Forward → MLP Inverse → MSE | Forward → FlowDiT inverse → flow loss | **Fixed** |
| Teacher | PhysicsPhaseLabeler (rules) | TeacherPhysicsLabeler | **Renamed** |
| Student | PhysicsPhaseRouter (sees z_tH, action at train) | StudentPhysicsRouter (z_t+text+state_t only) | **Fixed** |
| Physics → action | FiLM z_cond + head injection | physics_code → Forward + FlowDiT only | **Fixed** |
| Unified loss | Split idea2 / physics scripts | `compute_total_loss` | **Fixed** |

---

## Information Leakage Matrix

| Path | Pre-refactor | Post-refactor |
|------|--------------|---------------|
| z_tH → StudentRouter → FlowDiT | Leaked at train (zeroed at sim infer) | **Blocked** — Student never sees z_tH |
| z_tH → FlowDiT main path | Never | Never |
| z_tH → FlowDiT inverse/cycle | N/A (MLP inverse) | **Allowed** — explicit future_latent token |
| Teacher → action generation | Indirect via shared router | **Blocked** — Teacher only supervises L_phase |

---

## Module Mapping

| Target | File | Class |
|--------|------|-------|
| Future World Head | `models/world2wam_heads.py` | `ForwardHead` |
| Action Flow DiT | `models/action_dit.py` | `FlowActionDiT` |
| Light DiT (ablation) | `models/world2wam_heads.py` | `LightActionDiT` |
| MLP adapter (ablation) | `models/world2wam_heads.py` | `ActionAdapter` |
| Cycle helper | `models/world2wam_heads.py` | `compute_cycle_flow_loss` |
| Teacher | `physics/phase_labeler.py` | `TeacherPhysicsLabeler` |
| Student | `models/physics_router.py` | `StudentPhysicsRouter` |
| Wrapper | `models/physics_world2wam.py` | `PhysicsAlignedWorld2WAM` |
| Losses | `train/training_utils.py` | `compute_total_loss` |
| Physics losses | `train/physics_losses.py` | `compute_physics_losses` (L_phase + L_phy) |

---

## Train Scripts

| Script | Purpose |
|--------|---------|
| `train/train_world2wam_heads.py` | Stage1: ForwardHead (+ optional flow inverse if adapter present) |
| `train/train_action_flow_dit.py` | Stage2: FlowActionDiT + Forward + inverse/cycle flow |
| `train/train_physics_world2wam.py` | Full physics-aligned training with unified L_total |

---

## Deleted / Demoted

**Deleted:**
- `configs/world2wam_idea3_physics_spatial_h10.yaml` (v0)
- `docs/report_for_teacher_idea2_idea3_current.md`
- `scripts/inspect_physics_labels.py` (thin wrapper)
- v0 physics label main path in `physics_labels.py`

**Demoted to ablation:**
- `ActionAdapter` (MLP)
- `LightActionDiT`
- `InverseHead` (MLP, legacy checkpoint compat only)

**Out of scope (Version B):**
- Wan Video DiT, MoT, token-level ProPhy REB, neural Teacher, joint video-action flow

---

## Research Message

> We decouple world understanding and action generation in World Action Models. The world branch learns physics-aligned latent imagination, while the action branch uses flow-based generation to model multimodal robot actions. This enables stronger manipulation performance while preserving Fast-WAM's action-only low-latency inference.
