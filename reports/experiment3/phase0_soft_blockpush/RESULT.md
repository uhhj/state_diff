# Phase 0B Soft BlockPush result

## Verdict

- Probe-only: `PHASE0B_PROBE_MECHANISM_COMPLETE`.
- Full single Pair: `PHASE0B_SINGLE_PAIR_SCIENTIFIC_FAIL`.
- Failure cause: `translation_only_no_deformation_branch`.
- Scientific interpretation: hidden local friction produced an early formal sensor branch and a later absolute state/COM branch, but did not produce the required rigid-aligned non-rigid deformation branch.

## Verification

- Server repository: `/data/Experiment3/state_diff`.
- Python: `/miniforge3/envs/coord_bimanual/bin/python` (Python 3.9.15).
- Required pytest selection: 12 passed, 0 failed, 1 Gym precision warning.
- `pip check`: pre-existing environment conflict remains: `multiprocess 0.70.14` requires `dill>=0.3.6`, installed `dill` is `0.3.5.1`.
- Pairing checks: initial explicit state, fixed command arrays, absolute physics-step arrays, and phase arrays all matched.
- Node/visible counts: 72 / 24.
- Initial pusher-node signed distance: 2.1066 mm. This is recorded as a setup diagnostic; it exceeds the requested 2 mm diagnostic by 0.1066 mm but was not listed in the formal engineering gate. Probe contact and all formal engineering/no-action gates passed.

## Probe-only evidence

- No-action final/peak visible RMSE: 0.0000945 / 0.0000947 mm.
- Fused formal sensor onset: step 143 (`probe`).
- Oracle friction-load onset: step 145 (`probe`).
- Visible divergence onset: step 146 (`probe`).
- High/Free probe-window patch tangential force: 0.13366 / 0.00788 N.

## Default full Pair

- Final visible/full RMSE: 4.4510 / 4.4509 mm.
- Final COM gap: 5.4209 mm.
- Final rigid-aligned RMSE: 0.00211 mm (required: at least 1.0 mm).
- Absolute target-progress gap: 0.0588 mm (required: at least 5.0 mm).
- Branch amplification: 5.488.
- Structural edge ratio range: [0.98095, 1.00634].

## Allowed single-parameter trials

Three translation-only trials were executed; all retained independent configuration and reports.

| Trial | Sole physical change | Visible RMSE | COM gap | Rigid-aligned RMSE | Result |
|---|---|---:|---:|---:|---|
| D1 | patch x 0.418 → 0.4285 m | 15.451 mm | 18.915 mm | 0.00188 mm | translation only |
| D2 | test x 0.012 → 0.018 m | 8.468 mm | 11.361 mm | 0.00121 mm | translation only |
| D3 | shear force 0.35 → 0.25 N | 4.451 mm | 5.421 mm | 0.00211 mm | no measurable change |

The D sequence was exhausted. E-type trials were not used because the absolute test branch was already strong; increasing its magnitude would not address the missing rigid-aligned deformation mechanism.

Large trajectories, trace JSON, and MP4 files remain on the server under `/data/Experiment3/data/ccda_soft_blockpush_audit`. Generated plots and full reports remain under `/data/Experiment3/reports/phase0_soft_blockpush` and are intentionally not committed.
