# Phase3.14b-r2.5.6 Stage D Segment-Gate Recalibration

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r256_staged_physical_valid_branch_support_failed`
- Required next path: `REPAIR_CABLE_ONLY_BRANCH_TRANSPORT_WITH_FROZEN_RECALIBRATED_GATE`

## Frozen replay

- Isolated workers exact: `true`
- Model SHA256: `4a9ebcdb9d1995602a501c8820e07c0efcb757f1e3efff4ac0f2c5a4a23421d2`
- Reverse candidate SHA256: `1ab9f8d0026f215bf9e3b8aafeb4f8799bc1f8d329d8773e415413699a6c6d2d`

## Recalibrated gate

- Gate contract SHA256: `7dfe7881f3075354ccc38bc1e5ddfe3eb3b40f2c46c134bb3d0fab345b9ed03d`
- Joint robust-z threshold: `2543.16681`
- Calibration group acceptance: `0.964285714`
- Probe ground-truth row acceptance: `0.904761905`
- Probe ground-truth group acceptance: `0.916666667`
- Probe minimum condition acceptance: `0.857142857`

## Frozen predictions under recalibrated gate

- One-step t=10 segment acceptance: `0.849206349`
- Reverse segment row-any rate: `0.992063492`
- Reverse combined row-any rate: `0.984126984`
- Physical-valid branch eligible-row rate: `0.984126984`
- Physical-valid branch support among eligible: `0.717741935`

## Boundary

- The Stage-B model, scheduler, split, normalization, objective, seed and K were unchanged.
- Gate fitting used only grouped diagnostic-training ground truth.
- Probe targets and model candidates were not used to fit or select the gate.
- The condition label was not used to fit the gate.
- No checkpoint, tensor, candidate pool, NPZ, image or video was persisted.
- No formal diffusion/reverse, IDM, candidate execution, Phase4 or CPS was run.
- No action-diverse data was collected.
- `train_only_recommendation=None` and `selected_configuration=None`.
