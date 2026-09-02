# World2WAM ICLR 2027 experiment sprint

Updated: 2026-09-02

## Frozen claim

World2WAM uses bidirectional future-dynamics constraints and action-priority
gradient-conflict projection during training. The auxiliary future heads are
removed at export, so Version D adds no future rollout or auxiliary head to the
inference path. The paper tests whether this improves difficult-task and
out-of-distribution control robustness without adding inference-time future
imagination.

The claim is evaluated against the official FastWAM and Faster-WAM releases.
Tasks, seeds, initial states, episode counts, and statistical tests are frozen
before the remaining results are observed.

## Method matrix

All trainable variants use the same base checkpoint, data, optimizer settings,
LoRA rank 8/alpha 16, and 6,000 update steps.

| ID | Training objective | Gradient combination |
|---|---|---|
| R0 | Official FastWAM | none |
| B1 | Action-only LoRA | action only |
| B2 | B1 + forward future prediction | direct sum |
| B3 | B2 + inverse action reconstruction | direct sum |
| B4 | B3 + cycle consistency | direct sum |
| B5 / Version D | B4 | action-priority conflict projection |
| B5-no-hard | B5 with uniform RoboTwin sampling | conflict projection |

The default loss is

```text
L = 1.00 L_action + 0.10 L_forward + 0.05 L_inverse + 0.05 L_cycle.
```

RoboTwin B1--B5 use the same 70% difficult-sample sampling policy.
B5-no-hard is the causal control that separates the optimization method from
the sampler. LIBERO uses a fixed stratified cache of 12,000 clips: 3,000 clips
from each of Spatial, Object, Goal, and Long/10.

## Frozen experiments

### Standard LIBERO

- Suites: Spatial, Object, Goal, and Long/10.
- Ten tasks per suite and 50 fixed initial states per task.
- 2,000 episodes per checkpoint.
- Main comparison: FastWAM, Faster-WAM, and B5 seeds 42/43/44.
- B1--B5 seeds 42/43/44 form the ablation table.

### LIBERO-Plus

- Four suites and seven perturbation families: camera, robot initialization,
  language, lighting, background, noise, and layout.
- Fixed 15% stratified subset (approximately 1,505 task instances) for every
  required method; this is a protocol check and preliminary OOD table, not a
  result-dependent gate.
- Full 10,030-instance evaluation for FastWAM, Faster-WAM, and B5 seeds
  42/43/44.
- Report per-perturbation success, overall success, degradation from standard
  LIBERO, and the worst perturbation.

### RoboTwin

- Official 50-task list.
- clean2clean and clean2random.
- Ten matched episodes per task and condition.
- Main table: FastWAM, Faster-WAM, and B5 seeds 42/43/44.
- Causal table: B1--B5 and B5-no-hard, all three seeds.
- The historical hard-10 result remains a diagnostic result and is not the
  sole evidence for an overall gain.

### Efficiency and mechanism

- One A100, BF16, batch size one, common horizon and denoising steps.
- Twenty warm-up calls and 100 timed calls.
- Mean/P50/P95 latency, throughput, peak memory, parameters, checkpoint size,
  and training GPU-hours.
- Verify exported B5 actions match the pre-export policy.
- Log action/world gradient cosine, conflict rate, and projection norm every
  100 updates.
- Relate task-level conflict frequency to task-level success gain.
- Record steps-to-success, collision count where exposed by the simulator,
  first action difference, second-order jerk, and representative failures.

## Statistics and quality gates

- Every comparison uses matched task instances and initial states.
- Standard LIBERO and RoboTwin: stratified paired bootstrap 95% confidence
  intervals and train-seed mean plus standard deviation.
- LIBERO-Plus: suite/perturbation-stratified paired bootstrap and McNemar test.
- Report absolute gain, relative error reduction, worst ten tasks, and failure
  categories. Never select the best seed.
- Use “significantly improves” only when the paired 95% interval excludes zero.
- Pause a benchmark branch if the reproduced FastWAM Spatial result differs
  from 96.2% by more than two percentage points, while unrelated branches keep
  running.
- A formal result requires a frozen manifest, checkpoint hash, complete task
  denominator, no duplicate episode, and an explicit disposition for failures.

## Automated execution

`scripts/build_iclr2027_manifest.py` creates a 92-job dependency graph.
`scripts/paper_sprint.py` exposes `plan`, `run`, `resume`, `status`, and
`summarize`. The scheduler provides:

- GPU1--7 allow-listing and GPU0 exclusion;
- three consecutive idle checks, including compute-process checks;
- one lock per physical GPU;
- atomic job states and a frozen run manifest;
- 60/300/900-second retries;
- 500-step training checkpoints and artifact reconciliation after restart;
- 30-second launcher/installation supervision;
- a 200GB free-disk gate and independent continuation of unaffected branches.

The scheduler never terminates another user's process. If a card is occupied,
the job remains queued until it passes all idle checks.

## Execution order

1. Pin Faster-WAM and LIBERO-Plus; install isolated benchmark environments and
   obtain official checkpoints/assets.
2. Download all four LIBERO training suites and build four cache shards.
3. Train all missing LIBERO and RoboTwin ablations at seeds 42/43/44.
4. Run protocol smoke tests and the fixed LIBERO-Plus subset.
5. Run standard LIBERO and RoboTwin full tables.
6. Run the full LIBERO-Plus table.
7. Produce paired statistics, efficiency/mechanism analyses, figures, and
   LaTeX tables from the immutable raw results.

With seven continuously available A100s, the remaining complete chain is
expected to take roughly 5--9 days. Resource contention or simulator/download
failures can extend this to 8--14 days. Paper writing proceeds in parallel;
claims are updated only after the frozen comparisons complete.

## Repository policy

Git tracks source, configs, protocol, small summaries, and status documents.
It excludes model weights, datasets, credentials, caches, videos, and raw
machine-local logs. See `protocols/iclr2027_paper_sprint.json` for the
machine-readable protocol and `docs/ICLR2027_SPRINT_STATUS_20260902.md` for the
latest audited execution status.
