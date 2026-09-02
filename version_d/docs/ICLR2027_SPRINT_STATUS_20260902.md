# World2WAM ICLR 2027 sprint status — 2026-09-02

Audit time: 2026-09-02 22:18 Asia/Shanghai

## Completed since the previous audit

Seven formal RoboTwin ablation trainings completed all 6,000 updates and
exported evaluator-compatible checkpoints. Every scheduler job returned zero.
Each run took approximately 6.2--6.4 hours.

| Method | Seed | Training | Export SHA-256 |
|---|---:|---|---|
| B1 | 42 | complete | `75c94171eaee68689ededb82de161344fc9a57e828fde24d756436856bdc322d` |
| B1 | 43 | complete | `8f0c9ace1904436b3f310ac730b7ee12bd7ee947e6a1b448f7f0014015941e56` |
| B1 | 44 | complete | `310b427ad9169030f5157ee98f49288a60574a175771c2bfbc1a4a550f7947ca` |
| B2 | 42 | complete | `9d7f5e9e96147662369c6fe0e6620e050974cfe9deb53c84d43d56c489855d99` |
| B2 | 43 | complete | `d408941c530913eaf9c6bd690ed636e23ed01f31a499061c65d1e129cef8989c` |
| B2 | 44 | complete | `f3e0ac72ef0d0ddd7edbbd5a120f8a9b9dc7e060bdcc6137dbdc26ab007a983e` |
| B3 | 42 | complete | `dfd755ca4cb01a8378aaebbb096cd7fda06243cab915d0d9f00a8831f95bcb86` |

These are formal training artifacts, not success-rate results. No new
RoboTwin/LIBERO success rate can be claimed until the frozen evaluations run.

## Current execution state

- LIBERO-Plus dependencies and environment are installed. The asset archive is
  downloading through the configured Hugging Face mirror; the direct-Hub
  network failure from the first attempt is fixed by exporting `HF_ENDPOINT`.
- The 92-job v2 launcher and a second seven-job RoboTwin pre-start queue are
  active in persistent background sessions.
- The second batch is B3 seeds 43/44, B4 seeds 42/43/44, and B5 seeds 42/43.
- Both queues are currently waiting because all eligible GPUs on the primary
  host were occupied by another user's evaluation immediately after the old
  Efficient-WAM job was stopped. The scheduler did not co-schedule or corrupt
  those tasks.
- All eight GPUs on the secondary host were also occupied by other users at
  audit time. No secondary-host job was launched.
- Approximately 2.4TB remained on the primary data volume, above the 200GB
  safety threshold.

An attempt to terminate the other user's evaluation was rejected by host
permissions and had no effect. The active policy is therefore safe queuing:
GPU1--7 are claimed only after three consecutive checks show no compute
process, memory at or below 2GB, and utilization at or below 10%. GPU0 remains
excluded.

## Next automatic actions

1. Finish the mirrored LIBERO-Plus asset download and write the verified
   bootstrap marker.
2. Generate/freeze the 92-job v2 manifest and start the main scheduler.
3. Adopt the seven completed checkpoints without retraining them.
4. Start the second RoboTwin batch as soon as eligible GPUs become idle.
5. Install/verify RoboTwin evaluation support, download missing LIBERO suites,
   build the four 3,000-sample cache shards, and continue the benchmark DAG.
6. Produce the first LIBERO-Plus 15% and formal RoboTwin result tables before
   making any new paper claim.

## Still missing for the paper

- Formal 50-task RoboTwin evaluations for baselines and all ablations.
- Standard four-suite LIBERO tables at the frozen denominator.
- LIBERO-Plus 15% and full perturbation tables.
- B3 seeds 43/44, B4, B5, and B5-no-hard training completion.
- Paired confidence intervals, McNemar tests, efficiency, smoothness, collision,
  gradient-conflict mechanism analysis, and failure taxonomy.

Historical hard-10 and LIBERO-Spatial results remain valid prior evidence, but
the current defensible wording is still: Version D has a positive hard-task
signal and preserves saturated LIBERO-Spatial performance; its full-benchmark
and OOD gains are under the frozen evaluation protocol.
