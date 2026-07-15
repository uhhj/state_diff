# Phase3.14b-r2.5.6 Stage B Cable-only Diffusion Diagnostic

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r256_stageb_cable_only_branch_support_failed`
- Required next path: `REPAIR_CABLE_ONLY_BRANCH_TRANSPORT_BEFORE_FORMAL_PILOT`

## Determinism

- Isolated workers exact: `true`
- Final model SHA256: `4a9ebcdb9d1995602a501c8820e07c0efcb757f1e3efff4ac0f2c5a4a23421d2`

## Train-only diagnostic

- Train fixed-noise NMSE: `0.0039219358`
- Reverse best-of-K NMSE: `0.841358662`
- Best simple baseline NMSE: `2.47427535`
- Relative improvement: `0.659957546`
- Own-branch support rate: `0.738095238`
- Physical candidate rate: `0`

## Boundary

- Only the train/full-horizon state-v3 view was used.
- The target was `[4,48]` ordered cable XY; robot future was absent.
- No validation/formal-test target, checkpoint, weights, or prediction tensor was persisted.
- Reverse sampling was diagnostic only.
- No formal diffusion, IDM, candidate execution, Phase4, or CPS was run.
- Action-diverse IDM data was not collected in this stage.
- `train_only_recommendation=None` and `selected_configuration=None`.
