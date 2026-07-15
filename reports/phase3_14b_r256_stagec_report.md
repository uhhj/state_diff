# Phase3.14b-r2.5.6 Stage C Reverse Physical-Gate Attribution

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r256_stagec_segment_gate_self_calibration_failed`
- Required next path: `RECALIBRATE_FROZEN_CABLE_SEGMENT_GATE_BEFORE_MODEL_OR_BRANCH_REPAIR`

## Frozen Stage-B replay

- Isolated workers exact: `true`
- Model SHA256: `4a9ebcdb9d1995602a501c8820e07c0efcb757f1e3efff4ac0f2c5a4a23421d2`
- Reverse candidate SHA256: `1ab9f8d0026f215bf9e3b8aafeb4f8799bc1f8d329d8773e415413699a6c6d2d`
- Stage-B model, scheduler, split, thresholds and K were unchanged.

## Physical attribution

- Training-ground-truth segment candidate rate: `0.945080092`
- Probe-ground-truth segment candidate rate: `0.849206349`
- Reverse segment element pass rate: `0.726686508`
- Reverse segment candidate-all pass rate: `0`
- Reverse lower-bound violations: `8785`
- Reverse upper-bound violations: `16561`
- Reverse combined physical candidate rate: `0`

## Reverse trace: predicted-x0 segment candidate rate

- `t=99`: `0` (element `0.70198197`)
- `t=75`: `0` (element `0.724950397`)
- `t=50`: `0` (element `0.760093168`)
- `t=25`: `0` (element `0.779416839`)
- `t=10`: `0` (element `0.743087905`)
- `t=0`: `0` (element `0.726686508`)

## Branch attribution

- Historical K=8 support: `0.738095238`
- Pair-bootstrap 95% CI: `[0.6746031746031746, 0.7936507936507936]`
- Physical-valid-only branch audit available: `false`
- Historical classifier masked simultaneous physical failure: `true`

## Boundary

- Only the train/full-horizon view was used.
- No validation or formal-test target was opened.
- The Stage-B model and candidates were reproduced exactly.
- No model, scheduler, objective, threshold or candidate count was changed.
- No checkpoint, weights, candidate tensor, NPZ or video was persisted.
- No formal diffusion/reverse, IDM, candidate execution, Phase4 or CPS was run.
- No action-diverse data was collected.
- `train_only_recommendation=None` and `selected_configuration=None`.
