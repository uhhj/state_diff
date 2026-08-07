# CCDA Research Execution Plan V4.1: StateDiff Input and FT Baseline

## Status and scope

This is the active Experiment3 research plan. It supersedes the Phase 0B
material-calibration and deformation-gate amendments. Phase 0C validates hidden
dynamics and control relevance before any model training.

## Scientific contract

- The State Diff diffusion target is a 74D low-dimensional task state: 24
  top-layer deformable keypoints in XYZ followed by the end-effector XY.
- The formal contact sensor is a separate 45D causal condition: six motor
  torques, six joint reaction wrenches, and XYZ end-effector tracking error.
- Simulator contact points, friction coefficients, patch condition labels, and
  patch-contact statistics are privileged oracle data and never model inputs.
- Extended proprioception is an ablation input, not part of the 74D diffusion
  state.
- Visible deformation, strain, and curvature may be reported as diagnostics but
  are not Phase 0C hard gates.

## Phase 0C mechanics

The active HLF-SBP environment uses only the canonical Kelvin-Voigt material
`kv_ccda_v41`, an outer interval of 1/240 s, and eight true manual microsteps.
Internal force is recomputed before every microstep and Bullet `numSubSteps` is
one. No stiffness, angle, or microstep search is permitted.

## Hidden-dynamics evidence

The strict pair restores one common state, changes only the hidden local patch
friction, and executes byte-identical precomputed joint targets, end-effector
targets, and high-level actions. A same-condition repeat establishes the
deterministic floor. Completion requires a formal-sensor difference no later
than a future-state difference above the repeat/noise floor.

Control relevance is audited only after the strict pair passes. The fixed
straight, left-bias, and right-bias action library is evaluated under both
conditions without online selection.

## Required model baselines after validation

- B0 StateDiff and B1 StateDiff-FT use identical 74D future-state targets,
  backbone, inverse-dynamics model, train/validation split, and evaluation.
- B1 differs only by conditioning on the frozen 45D sensor history.
- CFPM/B2/B3 is not implemented until the direct B0-versus-B1 baseline exists.

No StateDiff, StateDiff-FT, CFPM, or IDM training occurs in Phase 0C.
