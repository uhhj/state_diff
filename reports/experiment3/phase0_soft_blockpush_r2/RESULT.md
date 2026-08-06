# Phase 0B-R2 true-microstep material calibration result

## Verdict

- Final verdict: `PHASE0B_R2_NO_PROFILE_PASSED`.
- Scientific status: true-manual-microstep mechanics converged, but both permitted axial material profiles were too stiff.
- Material frozen: no.
- Pair and training: not run and not authorized.

## Repository and preservation

- Starting SHA: `2620d25df8ba1f6dc823700a41d2baa632e711b1`.
- Branch: `Experiment3`.
- Frozen DeformableRavens gitlink: `282b93535b125d1a4487df85ad24aa41551957af`.
- Phase 0B and Phase 0B-R1 reports/evidence remain present.
- Legacy P2P and R1 Kelvin-Voigt regression tests remain in the required test selection.

## Verification

- Server repository: `/data/Experiment3/state_diff`.
- Environment: `/miniforge3/envs/coord_bimanual/bin/python`.
- Required legacy and R2 pytest selection: 24 passed, 0 failed, one existing Gym precision warning.
- `pip check`: only the known pre-existing `multiprocess 0.70.14` / `dill 0.3.5.1` conflict; dependencies were not changed.

## True microstep mechanics

- Outer timestep: `1/240 s`.
- Candidates: `M = 4, 8, 16, 32`.
- Bullet `numSubSteps`: `1`.
- Internal forces are vectorized and recomputed before every manual microstep.
- External coupon load is redistributed and reapplied before every manual microstep.
- Outer physical time is unchanged: `micro_dt = outer_dt / M`.
- Every full-block outer step contains `512 * M` force evaluations; energy records the final microstep while counts and maxima aggregate across microsteps.

## Mechanics validation

- Worst-case profile: `kv_r2_a8_z025`.
- All two-node and 2x2x2 runs were finite, zero-cap, structurally bounded, and passed their stability gates.
- `4 vs 8` did not converge because the two-node peak-extension and zero-crossing relative errors exceeded 5%.
- `8 vs 16` converged; selected `M = 8`, `micro_dt = 0.0005208333333333333 s`.
- Validation verdict: `PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE`.

## Profile trials

The two-profile maximum was respected.

1. A6 axial: `PHASE0B_R2_AXIAL_TOO_STIFF`; peak primary displacement `0.588406 mm`, peak lateral leakage effectively zero, recovery ratio `3.93e-14`, structural edge ratio `[0.989193, 1.012927]`, force-cap fraction `0`, anchor/no-action drift `0`.
2. A4 axial: `PHASE0B_R2_AXIAL_TOO_STIFF`; peak primary displacement `0.881438 mm`, peak lateral leakage effectively zero, recovery ratio `1.21e-13`, structural edge ratio `[0.983577, 1.019370]`, force-cap fraction `0`, anchor/no-action drift `0`.

Both responses were below the frozen axial lower gate of `2 mm`. Because A4 was the only permitted second profile, no shear coupon or table-settle audit was run. No frozen-material directory or frozen JSON was created.

Large trajectories, videos, plots, and complete JSON summaries remain on the server under:

- `/data/Experiment3/data/ccda_soft_blockpush_r2/`
- `/data/Experiment3/reports/phase0_soft_blockpush_r2/`

The next task must be a separately authorized revision of the bounded material-profile family followed by mechanics validation and the full axial-first calibration sequence. Pair execution and model training remain prohibited.
