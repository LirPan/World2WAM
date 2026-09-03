# World2WAM: Training-Time Future Dynamics for Robust World Action Models

Status: writing scaffold based on evidence available on 2026-09-03. Numbers marked
**preliminary** must not be presented as full-benchmark or statistically
significant results until the frozen evaluations finish.

## One-sentence thesis

World2WAM improves difficult-task and distribution-shift robustness by using
bidirectional future-dynamics supervision and action-priority gradient-conflict
projection during training, while removing the auxiliary heads at export so
that inference requires no future rollout.

## Claim ladder

### Supported now

1. The method has a positive signal on the fixed RoboTwin hard-10 diagnostic:
   FastWAM to Version D is 47% to 54% on clean initialization and 42% to 45%
   on randomized initialization.
2. Version D does not sacrifice saturated in-distribution LIBERO-Spatial
   performance. The best aligned variant reaches 489/500 (97.8%), compared
   with 481/500 (96.2%) for FastWAM.
3. The auxiliary future heads are training-only and are removed at export;
   the intended inference graph therefore has no future rollout overhead.

### Not yet supported

- Do not claim overall RoboTwin improvement until the paired 50-task table and
  confidence interval are complete.
- Do not claim cross-suite LIBERO improvement from Spatial alone.
- Do not claim OOD robustness until LIBERO-Plus perturbation results complete.
- Do not claim statistical significance or state of the art from the current
  preliminary numbers.

## Abstract scaffold

World Action Models jointly model visual dynamics and actions, but future
dynamics supervision can compete with the control objective, while explicit
future imagination at inference adds cost. We introduce World2WAM, a
training-time framework that combines forward prediction, inverse action
reconstruction, and cycle consistency with action-priority conflict
projection. When an auxiliary gradient conflicts with the action gradient,
its conflicting component is removed before the shared parameters are
updated. The auxiliary heads are discarded after training, preserving the
base policy's inference path. Preliminary results show positive gains on a
fixed RoboTwin hard-task diagnostic and no degradation on saturated
LIBERO-Spatial. We evaluate the final method under a frozen protocol on the
full RoboTwin suite, four standard LIBERO suites, and seven LIBERO-Plus shift
families, together with causal ablations, paired statistics, and efficiency
analysis. [Insert final full-benchmark and OOD numbers only after completion.]

## 1. Introduction

1. Motivate World Action Models: learned visual dynamics provide a useful
   representation for control, but action quality remains the actual target.
2. Identify the tension: auxiliary future prediction is helpful when aligned
   with action learning and harmful when its gradient competes with it.
3. Identify the deployment constraint: inference-time future rollout is
   expensive and undesirable for real-time control.
4. Present the central idea: use future information only during training and
   explicitly protect the action objective during shared-parameter updates.
5. Summarize evidence: difficult-task diagnostics, standard benchmarks, OOD
   perturbations, causal ablations, and efficiency.

### Planned contributions

- A training-only bidirectional future-dynamics objective for World Action
  Models, combining forward, inverse, and cycle constraints.
- Action-priority conflict projection that prevents auxiliary dynamics losses
  from degrading the control objective.
- A no-extra-rollout deployment path obtained by deleting auxiliary heads and
  merging LoRA weights at export.
- A frozen, paired evaluation across RoboTwin, LIBERO, and LIBERO-Plus with
  robustness, efficiency, and mechanism analyses.

## 2. Related Work

Organize around four themes rather than a long paper-by-paper catalogue:

1. Vision-language-action and diffusion/flow policies.
2. World models and World Action Models for robot control.
3. Auxiliary dynamics, inverse models, and cycle consistency.
4. Multi-objective optimization and gradient-conflict methods.

The positioning sentence should emphasize that World2WAM uses future dynamics
as privileged training supervision and protects action learning, rather than
adding an inference-time planner or rollout module.

## 3. Method

### 3.1 Problem setup

Define observation history, language instruction, action chunk, future visual
state, shared policy parameters, and the base FastWAM action objective.

### 3.2 Bidirectional future-dynamics constraints

Use the frozen loss weights:

```text
L = 1.00 L_action + 0.10 L_forward + 0.05 L_inverse + 0.05 L_cycle.
```

- `L_forward`: predict a future latent/visual representation from the current
  state and action.
- `L_inverse`: reconstruct the action from current and future representations.
- `L_cycle`: require the forward and inverse mappings to agree.

Explain that forward prediction asks whether the action explains the future,
while inverse reconstruction asks whether the future preserves
action-relevant information.

### 3.3 Action-priority conflict projection

Let `g_a` be the action gradient and `g_w` an auxiliary world gradient. If
their dot product is negative, use

```text
g_w_aligned = g_w - min(0, <g_w, g_a>) / (||g_a||^2 + epsilon) * g_a.
```

Then update shared parameters with `g_a + g_w_aligned`. State clearly which
parameters are shared, how multiple auxiliary gradients are combined, and
that the action gradient is never projected away.

### 3.4 Training and export

- LoRA rank 8, alpha 16; common 6,000-step budget.
- Same data, optimizer, batch size, and base checkpoint across ablations.
- RoboTwin uses a 70% difficult-sample policy; B5-no-hard isolates this effect.
- Auxiliary heads are removed and LoRA weights are merged for evaluation.
- Verify exported and pre-export action outputs numerically.

## 4. Experimental Setup

### 4.1 Benchmarks

- RoboTwin: 50 tasks, clean2clean and clean2random, ten paired episodes per
  task and condition.
- Standard LIBERO: Spatial, Object, Goal, and Long/10; 50 fixed episodes per
  task.
- LIBERO-Plus: seven shift families and both a fixed 15% protocol subset and
  the full 10,030-instance evaluation.

### 4.2 Baselines and ablations

- FastWAM and Faster-WAM.
- B1 Action-only; B2 +Forward; B3 +Inverse; B4 +Cycle with naive summation;
  B5/Version D with conflict projection; B5-no-hard.
- Seeds 42, 43, and 44; never select the best seed.

### 4.3 Metrics and statistics

Report success, paired 95% confidence intervals, seed mean and standard
deviation, relative error reduction, worst-task success, steps-to-success,
latency P50/P95, peak memory, collisions, action variation, jerk, checkpoint
size, and training GPU-hours. Use McNemar tests for matched LIBERO-Plus
instances and stratified paired bootstrap elsewhere.

## 5. Results

### 5.1 Preliminary evidence available now

| Benchmark | FastWAM | Version D | Delta | Scope |
|---|---:|---:|---:|---|
| RoboTwin hard-10 clean | 47% | 54% | +7 pp | preliminary diagnostic |
| RoboTwin hard-10 random | 42% | 45% | +3 pp | preliminary diagnostic |
| LIBERO-Spatial | 96.2% | 97.8% | +1.6 pp | 10 tasks x 50 episodes |

The current interpretation is a positive difficult-task signal and preserved
or improved performance in a saturated suite. It is not yet evidence of
uniform full-benchmark superiority.

### 5.2 Main RoboTwin table — pending

FastWAM, Faster-WAM, and Version D seeds 42/43/44 on all 50 tasks under both
conditions. Include the worst ten tasks and paired deltas.

### 5.3 Standard LIBERO table — pending

Four suites, common initial states, three Version D seeds, and both baselines.

### 5.4 LIBERO-Plus robustness — pending and paper-critical

Report each perturbation family, overall shift success, degradation from the
matching standard suite, and the worst perturbation. This table decides whether
the strongest final framing is general OOD robustness or difficult-task
optimization.

### 5.5 Causal ablation — training substantially complete, evaluation pending

Use B1 through B5 to show the incremental effect of forward, inverse, cycle,
and conflict projection. Use B5-no-hard to separate optimization from sampling.

### 5.6 Efficiency and mechanism — pending

Show that Version D retains the FastWAM inference path, then analyze gradient
cosine, conflict frequency, projection norm, and their task-level correlation
with success gains.

## 6. Discussion and Limitations

- Gains may concentrate on difficult or shifted tasks while saturated suites
  leave little headroom.
- Hard-sample selection and gradient projection must be causally separated.
- Simulation success does not establish real-robot robustness.
- Training adds auxiliary computation even though inference does not.
- Negative or neutral suites must be reported rather than hidden.

## 7. Conclusion

Restate the final supported claim at the strength permitted by the completed
confidence intervals. Avoid “state of the art” unless the frozen full tables
actually establish it.

## Planned figures and tables

1. Method diagram: shared backbone, F/I/C heads, action-priority projection,
   and stripped inference graph.
2. Main RoboTwin success table.
3. Standard LIBERO success table.
4. LIBERO-Plus seven-perturbation bar chart.
5. B1–B5 plus no-hard ablation table.
6. Efficiency table and latency/memory plot.
7. Gradient-cosine/conflict histogram and task-gain correlation.
8. Representative success/failure cases.

## Writing readiness

The Introduction, Related Work, Method, Experimental Setup, protocol, and
limitations can be written now. The abstract and conclusion should remain
number-light. Main claims, title wording, and Results conclusions should be
frozen only after the full RoboTwin and LIBERO-Plus paired evaluations finish.
