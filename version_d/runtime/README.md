# Version D runtime snapshot

This directory contains the Python source snapshot that produced the current
RoboTwin Version D checkpoints and evaluations on the yjh servers.

- Snapshot date: 2026-09-02
- Source host: `New_yjh` / `fiveages-A100-3`
- Server source root:
  `/DATA/disk0/yjh/robotwin_w2wam/latest/code/policy_lora/src`
- Cross-server verification: the core R3 files had identical SHA-256 hashes on
  `New_yjh` and `FiveAges_A100_2`.

The snapshot now includes the manifest-based stratified cache selection,
per-ablation CLI overrides, and resume-safe 6,000-step training entry points
used by the ICLR 2027 sprint. The runtime is an overlay for an external
Fast-WAM checkout; it does not vendor
Fast-WAM weights, RoboTwin/LIBERO datasets, simulator assets, or checkpoints.
Put `policy_lora/src` at the root of the server-side policy workspace and keep
the external paths in the YAML configuration valid.

## Paper method entry points

- `policy_lora/src/train/train_lora_fic_hardtask.py`: R3 training and
  action-prioritized gradient projection.
- `policy_lora/src/losses/world2wam_losses.py`: action, forward, inverse, and
  cycle losses.
- `policy_lora/src/models/future_latent_head.py`: predicts the cached
  next-step latent from the current hidden state and action.
- `policy_lora/src/models/inverse_action_head.py`: predicts the action from
  the current hidden state and a target or predicted future latent.
- `policy_lora/src/data/precompute_future_latents.py`: future-latent cache
  preparation.
- `policy_lora/src/wrappers/fastwam_wrapper.py`: Fast-WAM training adapter and
  hidden-token capture.
- `policy_lora/src/tools/export_libero_checkpoint.py`: merge/export path for
  LIBERO evaluation.

## Experimental branch

`train_lora_fic_r4.py` adds behavior anchoring. Its existing hard-10 result
uses only two episodes per task and is stored as exploratory evidence. It is
not the current paper default and must not be mixed with R3 standard-protocol
tables.
