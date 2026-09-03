# World2WAM ICLR 2027 sprint status — 2026-09-03

Audit window: 20:06–20:28 Asia/Shanghai.

## Outcome

The main queue was repaired from 46 dependency failures to zero permanent
failures, and all seven eligible A100s on New_yjh were reassigned to
World2WAM. FiveAges_A100_2 remained fully occupied by other users and had no
World2WAM environment, so no task was launched there.

## Training progress

- RoboTwin B1, B2, B3, and B4 are complete for seeds 42, 43, and 44: twelve
  formal 6,000-step training runs in total.
- Evaluator-compatible merged checkpoints for the newly completed B3/B4 runs
  are present under the server checkpoint directory and are approximately
  12.0GB each.
- B5 seeds 42/43/44 and B5-no-hard seed42 are running.
- The old Efficient-WAM stage-1 run was safely stopped after its step-76,000
  recovery checkpoint was verified, consistent with the frozen sprint plan.

## Repairs applied

1. Switched headless LIBERO rendering from unavailable OSMesa to verified EGL.
2. Reused the existing official Wan2.2 component directory instead of trying
   to download it from ModelScope inside benchmark environments.
3. Isolated standard LIBERO and LIBERO-Plus source paths to prevent the wrong
   package from being imported.
4. Reused the complete local RoboTwin assets and removed the installer's
   interactive path prompt.
5. Recognized the already-downloaded LeRobot v2 archives by validating
   `meta/tasks.jsonl`; incompatible v3 directories are retained rather than
   deleted.
6. Restored 46 jobs that had been irreversibly marked failed only because an
   upstream download timed out.
7. Added external-running task adoption so a restarted supervisor does not
   duplicate jobs launched by the pre-start queue.
8. Removed two duplicate B5 processes before they could concurrently write the
   same checkpoint directories.

## Active allocation after repair

GPU0 remains excluded because it hosts another user's VLLM service.

| GPU | World2WAM work |
|---:|---|
| 1 | RoboTwin B5 seed43 |
| 2 | LIBERO-Plus 15% FastWAM |
| 3 | LIBERO-Plus 15% exploratory Version D |
| 4 | RoboTwin B5 seed42 |
| 5 | RoboTwin B5 seed44 |
| 6 | RoboTwin B5-no-hard seed42 |
| 7 | LIBERO-Plus 15% Faster-WAM |

The three LIBERO-Plus evaluators passed task selection, EGL initialization,
and the earlier missing-component failure. At the end of the audit they were
loading the local Wan2.2 model and had not yet produced success-rate results.

## Evidence available for writing

- RoboTwin fixed hard-10 diagnostic: FastWAM to Version D is 47% to 54% clean
  and 42% to 45% random.
- LIBERO-Spatial: FastWAM is 481/500 (96.2%) and the aligned Version D variant
  is 489/500 (97.8%).

These support a preliminary difficult-task signal and preservation or
improvement on a saturated suite. They do not yet support a full-benchmark,
OOD, statistically significant, or state-of-the-art claim. The paper scaffold
is in `docs/ICLR2027_PAPER_OUTLINE.md`.
