# Phase3.14b-r2.5.6 Stage D.2 Cable-XY Collapse Provenance Audit

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r256_staged2_xy_projection_collapse_confirmed`
- Required next path: `FREEZE_ONE_SIDED_XY_UPPER_SEGMENT_CONTRACT_AND_REAUDIT_FROZEN_REVERSE`

## Exact source reconstruction

- Isolated workers exact: `true`
- Raw source episodes opened: `384`
- Raw XY to cached target exact rate: `1`
- Raw XYZ audit SHA256: `be198be6c99abb83a204adcf3d1a6162aaa4aa7f35038c6e52acd3ac9e5c5c4a`
- Bead order stable: `true`

## Stage-D.1 driver

- Window-contract row index: `494`
- Raw pickle frame index: `3`
- Source: `raw/train/free/seed_400106.pkl`
- Horizon / segment: `0 / 11`
- XY reference ratio: `0.00124600759`
- XYZ reference ratio: `0.999178552`
- XY / XYZ projection ratio: `0.00124701698`
- Driver category: `xy_projection_artifact`

## Calibration lower-tail attribution

- Severe XY events: `21`
- Projection-artifact fraction: `1`
- True-XYZ-collapse fraction: `0`
- Partial-XYZ-collapse fraction: `0`

## Paired counterpart

- Severe event instances: `78`
- Both-condition severe fraction: `0.0769230769`
- Counterpart-not-severe fraction: `0.923076923`

## Boundary

- The raw pickles were SHA-verified and opened read-only.
- `row_index` was treated as the window-contract row, not a pickle frame.
- Raw frame mapping used `window_t + 1 + future_horizon`.
- Raw ordered XYZ was used for audit only; state-v3 and caches were not modified.
- DeformableRavens source was statically audited but not executed or modified.
- No gate was selected and no threshold was changed.
- No model training, reverse sampling, formal pilot, IDM, data collection, candidate execution, Phase4, or CPS was run.
- No checkpoint, tensor, NPZ, cache, image, or video was written.
- `train_only_recommendation=None` and `selected_configuration=None`.
