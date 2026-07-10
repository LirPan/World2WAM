# Version B — Physics-Aligned MoT World2WAM

Full FastWAM-style stack with physics routing on **MoT action tokens**.

## Architecture

```
LIBERO batch
  → FastWAMMotAdapter.training_forward()
       ├── MoT mixed attention (video expert + action expert)
       ├── L_fastwam_action (+ optional L_fastwam_video)
       └── hook: action_tokens [B, T, hidden_dim]

  → TokenStudentPhysicsRouter(h_t, text, proprio) → physics_code
  → TeacherMoTPhysicsLabeler (train only) → phase labels

  → MotWorldHeads
       ├── ForwardWorldHead(h_t, action, physics) → z_future
       ├── MotFlowActionExpert — main flow loss (physics, no future)
       └── MotFlowActionExpert — inverse/cycle (future_latent token)

  → L_total = L_fastwam + L_flow + L_future + L_inverse + L_cycle + L_phase + L_phy
```

## vs Version A (`minimal_world2wam`)

| | Version A | Version B |
|---|-----------|-----------|
| Backbone | Frozen FastWAM VAE encoder | **Full FastWAM MoT** (frozen by default) |
| State `h_t` | 48-d pooled VAE latent | **MoT action tokens** (pooled `[B, hidden_dim]`) |
| Physics router | `StudentPhysicsRouter(z_t, text, state)` | `TokenStudentPhysicsRouter(h_t, text, proprio)` |
| Action expert | Standalone `FlowActionDiT` | FastWAM `ActionDiT` in MoT + auxiliary `MotFlowActionExpert` |
| Video branch | Not trained | Optional `λ_video` joint training |

## Layout

```
version_b/
├── README.md
├── configs/
│   ├── default.yaml
│   ├── physics_mot_train.yaml
│   ├── bidirectional_mot_train.yaml
│   └── precompute_latents.yaml
├── scripts/
│   ├── 01_precompute_future_latents.py
│   ├── train_physics_mot.py
│   └── train_bidirectional.py
├── tests/
└── world2wam_vb/
    ├── adapters/       # FastWAM MoT loader + hook
    ├── data/           # LIBERO batch + latent cache
    ├── models/         # Physics MoT model + flow expert + heads
    ├── physics/        # Teacher labels + losses
    ├── losses/         # Bidirectional + unified total
    └── utils/
```

## Training pipeline

```bash
# 1) Precompute current + future VAE latents (GPU + FastWAM)
python scripts/01_precompute_future_latents.py --config configs/precompute_latents.yaml

# 2a) Bidirectional MoT baseline (no physics)
python scripts/train_bidirectional.py --config configs/bidirectional_mot_train.yaml

# 2b) Full physics-aligned MoT
python scripts/train_physics_mot.py --config configs/physics_mot_train.yaml
```

## Quick start (not run by default)

```bash
export PY=python
cd /DATA/disk0/jianhua/Physics-Aligned\ World2WAM/version_b

# Shape / leakage unit tests (no FastWAM GPU)
$PY tests/test_version_b_shapes.py

# Full training (requires FastWAM + LIBERO + GPU)
$PY scripts/train_physics_mot.py --config configs/physics_mot_train.yaml
```

## Inference

Physics modules are **train-only**. Deployment uses official `FastWAM.infer_action()` unchanged.
