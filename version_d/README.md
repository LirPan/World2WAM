# World2WAM Version D

Version D records the current RoboTwin improvement method and its reproducible evaluation entry points.

## What changed

The official FastWAM checkpoint remains the frozen R0 baseline. Version D adds a parameter-efficient LoRA adaptation and a small future-latent prediction branch:

1. LoRA is inserted into the attention projections `q/k/v/o` with rank 8 and alpha 16.
2. The action imitation loss remains the main objective.
3. A future latent is predicted from the current representation. Forward, inverse, and cycle consistency losses regularize the representation.
4. The auxiliary world-model gradients use conflict projection (`project_conflicts`) before they are combined with the action gradient.
5. Sampling emphasizes hard-task keywords (`dual`, `three`, `stapler`, `hammer`, `cabinet`, `switch`, `stamp`) at 70% of training samples.

The current loss weights are:

```text
L = 1.00 L_action + 0.10 L_forward + 0.05 L_inverse + 0.05 L_cycle
```

## Current validation result

The latest paired validation contains four groups, each with 5 tasks and 10 episodes per method (5 clean + 5 random per task). It is a validation matrix, not the full RoboTwin benchmark.

| Group | R0 clean | R3 clean | R0 random | R3 random | R0 avg | R3 avg |
|---|---:|---:|---:|---:|---:|---:|
| fixed5 | 72% | 72% | 60% | 64% | 66% | 68% |
| next5B | 68% | 68% | 72% | 70% | 70% | 69% |
| next5C | 66% | 70% | 64% | 66% | 65% | 68% |
| next5D | 64% | 66% | 62% | 56% | 63% | 61% |
| **four-group mean** | **67.5%** | **69.0%** | **64.5%** | **64.0%** | **66.0%** | **66.5%** |

Relative to R0, R3 is +1.5 percentage points on clean episodes, -0.5 points on random episodes, and +0.5 points overall. These numbers should be reported as validation evidence only until the full benchmark is complete.

## Reproduction entry points

- `configs/robotwin_r3_lora_fic_projection.yaml`: training and data configuration.
- `scripts/new_yjh_robotwin_supervisor.sh`: non-invasive server supervisor for environment setup, asset preparation, cache generation, training, export, and evaluation.
- `tools/eval_robotwin_physical.py`: physical RoboTwin evaluation wrapper.
- `tools/audit_robotwin_tasks.py`: task-set audit and hard-task selection evidence.
- `tools/summarize_matrix.py`: converts evaluator `summary.json` files into a comparison CSV.
- `docs/teacher_explanation.docx`: teacher-facing explanation of the method and current results.

The paths in the YAML and supervisor intentionally match the current yjh server layout. Change only the server-root variables and data/checkpoint paths when porting to another machine. Do not commit checkpoints, datasets, credentials, or server handoff notes.

## LIBERO Version D adaptation

`configs/libero_version_d_fiveages.yaml` is the LIBERO-specific 7D recipe. It
keeps the F/I/C losses and `project_conflicts` gradient merge, while using the
official LIBERO 2-camera/224 FastWAM interface. It is intentionally separate
from the RoboTwin 14D checkpoint and must be trained/exported before LIBERO
simulation evaluation.

After the RobotWin supervisor writes `status/robotwin_complete`, run
`scripts/libero_version_d_after_robotwin.sh`. The script precomputes the
future-latent cache, trains and exports Version D, evaluates official FastWAM
and Version D on the same fixed LIBERO-Spatial task IDs and seeds, then writes
`libero_pair_summary.json`. It does not select tasks, seeds, or failed trials
after seeing results.

If a mirror requires a SOCKS5 proxy, set `ROBOTWIN_SOCKS_PROXY` in the server environment; the supervisor defaults to a local proxy and does not embed an internal host address.

## Evaluation protocol

R0 and R3 must use the same task list, episode count, clean/random split, simulator settings, and evaluator. Preserve each evaluator output directory, especially `summary.json`, together with the command and return code. Smoke tests with three episodes are useful for debugging but must not be mixed with the paired validation table.
