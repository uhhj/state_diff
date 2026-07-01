# Phase 0 DeformableRavens Setup Report

## Summary

- Task: `cable-line-notarget`
- DeformableRavens root: `/data/state_diff2/external/deformable-ravens`
- Data episodes: `10`
- Goal episodes: `20`
- JSON summary: `/data/state_diff2/reports/phase0_defravens_summary.json`

## Pass Criteria

- DeformableRavens import smoke test passes.
- `cable-line-notarget` is present in `tasks.names`.
- `data/cable-line-notarget` has at least 10 smoke demos.
- `goals/cable-line-notarget` has at least 20 goals.
- `reports/phase0_defravens_summary.json` exists.
- `reports/phase0_defravens_setup.md` exists.
- Visualization videos or images are not required for Phase0.
- Dataset fields include `color`, `depth`, `action`, `info`, `last_color`, `last_depth`, `last_info`.

## Data Field Counts

### data
- `color`: exists=True, files=10, first=000000-7.pkl
- `depth`: exists=True, files=10, first=000000-7.pkl
- `action`: exists=True, files=10, first=000000-7.pkl
- `info`: exists=True, files=10, first=000000-7.pkl
- `last_color`: exists=True, files=10, first=000000-7.pkl
- `last_depth`: exists=True, files=10, first=000000-7.pkl
- `last_info`: exists=True, files=10, first=000000-7.pkl

### goals
- `color`: exists=True, files=20, first=000000-6.pkl
- `depth`: exists=True, files=20, first=000000-6.pkl
- `action`: exists=True, files=20, first=000000-6.pkl
- `info`: exists=True, files=20, first=000000-6.pkl
- `last_color`: exists=True, files=20, first=000000-6.pkl
- `last_depth`: exists=True, files=20, first=000000-6.pkl
- `last_info`: exists=True, files=20, first=000000-6.pkl

## Notes

This is Phase0 only. No hidden contact condition is added here.
Phase1 should fork or subclass the cable task after this original-task smoke test passes.
