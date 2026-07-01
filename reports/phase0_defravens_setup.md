# Phase 0 DeformableRavens Setup Report

## Summary

- Task: `cable-line-notarget`
- DeformableRavens root: `/data/state_diff2/external/deformable-ravens`
- Data episodes: `10`
- Goal episodes: `20`
- JSON summary: `/data/state_diff2/reports/phase0_defravens_summary.json`

## Pass Criteria

- `cable-line-notarget` import smoke test passes.
- `data/cable-line-notarget` has 10 smoke demos.
- `goals/cable-line-notarget` has 20 goals.
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
- `color`: exists=True, files=20, first=000000-5.pkl
- `depth`: exists=True, files=20, first=000000-5.pkl
- `action`: exists=True, files=20, first=000000-5.pkl
- `info`: exists=True, files=20, first=000000-5.pkl
- `last_color`: exists=True, files=20, first=000000-5.pkl
- `last_depth`: exists=True, files=20, first=000000-5.pkl
- `last_info`: exists=True, files=20, first=000000-5.pkl

## Notes

This is Phase0 only. No hidden contact condition is added here.
Phase1 should fork or subclass the cable task after this original-task smoke test passes.

## Visualization Outputs

- Output directory: `/data/state_diff2/reports/phase0_visualizations/cable-line-notarget`
- Episodes requested: `3`
- Camera index: `0`
- FPS: `12`
- Hold seconds per high-level timestep: `0.6`
- Minimum video duration seconds: `5.0`
- Color videos: `3`
- Depth videos: `3`
- Overview PNG: `/data/state_diff2/reports/phase0_visualizations/cable-line-notarget/overview_contact_sheet.png`

### Warnings
- None for selected episodes.

### Files
- `/data/state_diff2/reports/phase0_visualizations/cable-line-notarget/demo_000000_color.mp4`
- `/data/state_diff2/reports/phase0_visualizations/cable-line-notarget/demo_000001_color.mp4`
- `/data/state_diff2/reports/phase0_visualizations/cable-line-notarget/demo_000002_color.mp4`
- `/data/state_diff2/reports/phase0_visualizations/cable-line-notarget/demo_000000_depth.mp4`
- `/data/state_diff2/reports/phase0_visualizations/cable-line-notarget/demo_000001_depth.mp4`
- `/data/state_diff2/reports/phase0_visualizations/cable-line-notarget/demo_000002_depth.mp4`
- `/data/state_diff2/reports/phase0_visualizations/cable-line-notarget/overview_contact_sheet.png`
