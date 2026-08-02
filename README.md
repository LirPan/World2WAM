# Physics-Aligned World2WAM

**Learn the world with physics, generate actions with diffusion.**

> **Branch guide for advisors:** see [`VERSIONS.md`](VERSIONS.md).  
> - [`version-c`](https://github.com/LirPan/World2WAM/tree/version-c) — physics-gated residual  
> - [`latest`](https://github.com/LirPan/World2WAM/tree/latest) — ICLR ActionDiT-LoRA pilot (this branch when on `latest`)  
> - Results: [`docs/ICLR_PILOT_RESULTS.md`](docs/ICLR_PILOT_RESULTS.md)

Version A architecture: decouple world understanding from action generation on frozen FastWAM pooled latents.

## Architecture

- **Future World Head** (`ForwardHead`): `z_t + action + text + physics → z_hat_future`
- **Action Flow DiT** (`FlowActionDiT`): main decoder — `z_t + text + physics → action` (no future latent)
- **Inverse / Cycle**: shared `FlowActionDiT` with `future_latent` token (train only)
- **TeacherPhysicsLabeler**: rule-based phase pseudo-labels (train only)
- **StudentPhysicsRouter**: `z_t + text + state_t → physics_code` (train + infer)

## Train scripts

| Script | Stage |
|--------|-------|
| `train/train_world2wam_heads.py` | Stage 1: ForwardHead |
| `train/train_action_flow_dit.py` | Stage 2: FlowActionDiT + forward/inverse/cycle |
| `train/train_physics_world2wam.py` | Physics-aligned full model |

`train/train_world2wam_adapter.py` is a backward-compatible wrapper for Stage 2.

## Configs

| Config | Purpose |
|--------|---------|
| `configs/world2wam_libero_spatial_h10_paper.yaml` | Stage 1 paper baseline |
| `configs/world2wam_libero_spatial_h10_flow_dit.yaml` | Stage 2 FlowDiT |
| `configs/world2wam_physics_flow_dit_main.yaml` | **Main experiment** (Physics + FlowDiT) |
| `configs/world2wam_libero_spatial_h10_physics_v1.yaml` | Ablation: MLP adapter + physics |
| `configs/world2wam_libero_spatial_h10_dit.yaml` | Ablation: LightActionDiT |

## Eval

- Offline: `eval/eval_offline_cache_only.py`
- LIBERO sim: `eval/eval_libero_world2wam.py` (`baseline`, `ours_onestep_*`, `ours_onestep_physics_flow_dit`, residual modes)

## Docs

- [docs/final_architecture_audit.md](docs/final_architecture_audit.md) — gap analysis and module map
- [docs/gpt_handoff_world2wam_full_status.md](docs/gpt_handoff_world2wam_full_status.md) — historical status

## Symlink

`minimal_world2wam` → `Physics-Aligned World2WAM` at workspace root for import compatibility.

## Version B (MoT stack)

Full FastWAM MoT + physics routing on action tokens. Code under [`version_b/`](version_b/).

| Script | Purpose |
|--------|---------|
| `version_b/scripts/01_precompute_future_latents.py` | Precompute current + future VAE latents |
| `version_b/scripts/train_physics_mot.py` | Physics-aligned MoT training |
| `version_b/scripts/train_bidirectional.py` | Bidirectional MoT baseline (no physics) |
| `version_b/scripts/poll_gpu_version_b.sh` | GPU polling launcher |

See [version_b/README.md](version_b/README.md).

## Version C (Physics-Gated Residual)

Same world losses as Version A (Forward / Inverse / Cycle / Physics), but Flow targets \(\delta=a_{\text{GT}}-a_{\text{FW}}\) and LIBERO uses \(a=a_{\text{FW}}+\alpha\cdot\hat\delta\).

| Item | Path |
|------|------|
| Config | `configs/world2wam_physics_residual_flow_dit_vc.yaml` |
| Pipeline | `scripts/run_version_c_pipeline.sh` |
| Docs | [docs/version_c_residual.md](docs/version_c_residual.md) |

```bash
bash scripts/run_version_c_pipeline.sh all
```

## Latest (ICLR): ActionDiT-LoRA + world regularizers

Primary **control** path for beating FastWAM’s ~96.2% LIBERO-Spatial floor: PEFT LoRA on ActionDiT, optional Forward/Inverse/Cycle regularizers, official `eval_libero_single` protocol.

| Item | Path |
|------|------|
| Package | [`policy_lora/`](policy_lora/) |
| Pilot results | [docs/ICLR_PILOT_RESULTS.md](docs/ICLR_PILOT_RESULTS.md) |
| Train action-LoRA | `bash scripts/train_policy_lora_action.sh` |
| Train LoRA+FIC | `bash scripts/train_policy_lora_fic.sh` |
| Export merged ckpt | `bash scripts/export_lora_fic_official.sh` |

Best pilot numbers (seed 42): **B5 overall 97.8% (489/500)**; **B1 hard 97.6% (244/250)**. Floor: 96.2% / hard 234/250.

## Background jobs (GPU poll)

```bash
# Version A full pipeline (precompute → train → LIBERO eval)
bash scripts/poll_gpu_version_a.sh start
bash scripts/poll_gpu_version_a.sh status

# Version B
bash version_b/scripts/poll_gpu_version_b.sh start
```

## External dependencies (not in repo)

- FastWAM official checkpoint + Wan VAE weights (~62 GB)
- LIBERO spatial LeRobot dataset
- Precomputed latent cache (`cache/libero_spatial_h10_full_fastwam/`)

See `scripts/setup_deps.sh` and `scripts/migrate_to_remote.sh` for deployment.
