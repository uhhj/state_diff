# PB2-A/B DLO-Lab Wrapping Reproduction & Winding Audit

Verdict: `PB2AB_WRAPPING_REPRODUCED_AND_WINDING_AUDITED`

## PB2-A — official reproduction

- Official CMA-ES last iter: 50
- Official best reward: 161.504562
- best_traj shape: `(10, 12)`
- best_qpos shape: `(101, 18)`
- Replay finite: True

## PB2-B — task-native privileged winding

- Official/reconstructed winding-loss max abs diff: 0.000e+00
- Dynamic variation observed: True
- Maximum signed-turn span: 1.0000004

Per-post signed and magnitude ranges are stored in `EVIDENCE.json`.

## Findings

- `PB2A_OFFICIAL_WRAPPING_REPRODUCED`
- `PB2B_TASK_NATIVE_WINDING_DEFINITION_REPRODUCED`
- `PB2B_TASK_SEMANTIC_WINDING_TRANSITION_OBSERVED`

## Next action

Proceed to PB2-C natural pair discovery. Use the published Wrapping task unchanged, keep winding as privileged audit metadata, define deployable partial observation separately, and use repository-provided Wrapping position randomization for rollout diversity without changing task physics. Do not start StateDiff training yet.

## Boundary

This phase stops before natural pair mining, snapshot branching, StateDiff training, CFPM, IDM, or closed-loop evaluation.
