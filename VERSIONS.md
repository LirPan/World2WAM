# World2WAM — Branch Guide (for collaborators / advisors)

Repository: [https://github.com/LirPan/World2WAM](https://github.com/LirPan/World2WAM)

| Branch | What it is | Read first |
|--------|------------|------------|
| [`main`](https://github.com/LirPan/World2WAM/tree/main) | Snapshot of Version A + Version B (physics FlowDiT / MoT) | Historical baseline code |
| [`version-c`](https://github.com/LirPan/World2WAM/tree/version-c) | **Version C**: physics-gated residual \(\delta = a_{\mathrm{GT}}-a_{\mathrm{FW}}\), \(a=a_{\mathrm{FW}}+\alpha_{\mathrm{eff}}\hat\delta\) + LIBERO eval parity | [docs/version_c_residual.md](docs/version_c_residual.md) |
| [`latest`](https://github.com/LirPan/World2WAM/tree/latest) | **ICLR control path**: ActionDiT-LoRA + world regularizers (B1–B5 pilot) on top of Version C | [docs/ICLR_PILOT_RESULTS.md](docs/ICLR_PILOT_RESULTS.md) |

## Recommended reading order for a progress report

1. This file (`VERSIONS.md`)
2. Branch **`latest`** → [docs/ICLR_PILOT_RESULTS.md](docs/ICLR_PILOT_RESULTS.md) (success-rate table)
3. Branch **`version-c`** → residual interface + why \(\alpha=0\) must recover FastWAM
4. Optional: Version A/B under `main` for world-model architecture context

## Absolute links

- Version C: https://github.com/LirPan/World2WAM/tree/version-c
- Latest (ICLR LoRA): https://github.com/LirPan/World2WAM/tree/latest
- Main (A/B): https://github.com/LirPan/World2WAM/tree/main

## What is intentionally not in git

- Large `.pt` checkpoints / merged FastWAM weights (~12 GB)
- Latent caches (`cache/libero_spatial_h10_full_fastwam/`)
- Per-trial LIBERO rollout logs
