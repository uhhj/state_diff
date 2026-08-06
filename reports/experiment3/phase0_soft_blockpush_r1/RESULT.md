# Phase 0B-R1 compliant soft-block result

## Verdict

- Final verdict: `PHASE0B_R1_COUPON_ENGINEERING_BLOCKED`.
- Scientific status: no material profile passed the mandatory coupon, so no R1 Pair was run.
- Model training: not allowed.

Legacy Phase 0B code, configuration, committed evidence, and `p2p_legacy` mechanics remain intact. R1 adds a 72-node explicit Kelvin–Voigt model with 162 structural, 242 shear, and 108 bending springs and no internal P2P constraints. Equal-and-opposite spring/damper forces are aggregated per node before each 240 Hz outer Bullet step. Two deterministic Bullet internal integration substeps stabilize the C0 explicit-force integration without changing the outer trace rate, solver iterations, or material coefficients.

## Verification

- Server repository: `/data/Experiment3/state_diff`.
- Environment: `/miniforge3/envs/coord_bimanual/bin/python`.
- Required legacy and R1 pytest selection: 19 passed, 0 failed, one existing Gym precision warning.
- `pip check`: the known pre-existing `multiprocess 0.70.14` / `dill 0.3.5.1` conflict is retained; no dependency was changed for R1.

## Coupon C0 — nominal

- Verdict: `PHASE0B_R1_COUPON_UNSTABLE`.
- Peak rigid-aligned RMSE: 3.170 mm (within the 1–8 mm gate).
- Peak face-relative displacement: 47.486 mm (above the 12 mm maximum).
- Peak structural/shear strain RMS: 0.1591 / 0.1563.
- Structural edge ratio range: [0.3330, 1.6685].
- No-action visible drift: 0.693 mm.
- Force-cap fraction: 0.
- Recovery ratio: 0.2073.

Because C0 was unstable rather than too rigid, the instructions permitted C2 and prohibited C1.

## Coupon C2 — stiff

- Verdict: `PHASE0B_R1_COUPON_ENGINEERING_BLOCKED`.
- Peak rigid-aligned RMSE: 781.633 mm.
- Peak face-relative displacement: 348.102 mm.
- Peak structural/shear strain RMS: 191.114 / 136.643.
- Structural edge ratio range: [0.0129, 365.024].
- Force-cap fraction: 0.7998.
- Recovery ratio: 0.9067.
- Engineering failure: floor penetration and spring runaway.

The two-profile maximum was reached without a passing coupon. Consequently no material profile was frozen, and the strict coupon gate prohibited probe-only, full Pair, geometry trials, development seeds, and training.

Large coupon trajectories, trace JSON, videos, and plots remain on the server under:

- `/data/Experiment3/data/ccda_soft_blockpush_r1/coupon/`
- `/data/Experiment3/reports/phase0_soft_blockpush_r1/coupon/`

The next permitted task is limited to revising and independently validating the material/coupon mechanics. Patch, pusher, Pair motion, and model training remain out of scope until a coupon passes.
