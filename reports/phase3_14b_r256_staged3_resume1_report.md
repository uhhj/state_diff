# Phase3.14b-r2.5.6 Stage D.3 Resume1

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r256_staged3_cable_x0_upper_segment_expansion_failed`
- Required next path: `ADD_ORDERED_SEGMENT_UPPER_EXPANSION_OBJECTIVE_TO_CABLE_X0_DENOISER`

## Blocked-attempt provenance

- Original implementation commit: `7713f9feb07d2053744dc67819cd59682c03370a`
- Blocked evidence commit: `cff540490bd90ea3cb261bf13fbf63bb5991d051`
- Original test-gate SHA256: `210bd16acc348bfa1d90df016083be490fe9f28b588e5ab9cd80b42c956a9c9b`
- Original blocked-summary SHA256: `dd53e4170ed8d33f29fcbb3e425c41c3e21ee57cc138ffabb469a029c90066cd`
- Correction: `raw_xy_float32_exact_required -> cache_float32_exact_required`

## Frozen contract

- Gate selected: `true`
- Contract SHA256: `24f5ce7aa5c6d3c45ab6d4e659fa1c4719ffb486b61022db453b289e990b2781`
- Upper threshold: `3.937946394`
- Threshold bit-exact with Stage-D upper: `true`
- Calibration group acceptance: `0.964285714`
- Lower score role: `diagnostic_only`

## Frozen replay

- Isolated workers exact: `true`
- Model SHA256: `4a9ebcdb9d1995602a501c8820e07c0efcb757f1e3efff4ac0f2c5a4a23421d2`
- Reverse candidate SHA256: `1ab9f8d0026f215bf9e3b8aafeb4f8799bc1f8d329d8773e415413699a6c6d2d`

## Ground truth

- Probe upper row acceptance: `0.992063492`
- Probe upper group acceptance: `0.958333333`
- Probe minimum condition acceptance: `0.984126984`

## Frozen predictions

- One-step t=10 upper row acceptance: `0`
- One-step t=25 upper row acceptance: `0`
- One-step t=50 upper row acceptance: `0`
- Reverse upper candidate / row-any: `0 / 0`
- Reverse combined candidate / row-any: `0 / 0`

## Physical-valid branch

- K=8 eligible-row rate: `0`
- K=8 support among eligible: `unavailable`

## Boundary

- The original failed Stage-D.3 test gate and blocked summary remain immutable and committed.
- Resume1 changes only the D.2 static source-predicate signature.
- The selected gate is upper-only grouped split-conformal XY validity.
- The threshold was recomputed and matched the historical Stage-D upper threshold bit-exactly.
- The Stage-B model, scheduler, split, normalization, objective, noise and K were unchanged.
- Historical D/D.1/D.2 and failed D.3 files were not modified.
- No formal pilot, formal diffusion/reverse, IDM, data collection, candidate execution, Phase4 or CPS was run.
- No checkpoint, tensor, NPZ, cache, image or video was persisted.
- `train_only_recommendation=None` and `selected_configuration=None`.
