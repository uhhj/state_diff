# Phase 0J — Outcome Localization and Fixed Wide-Stop Z-Latch Smoke

- Topology: `wide_stop_z_latch_v2` (single fixed topology; no grid search)
- Only physical change: `latch.wall_width 0.032 m -> 0.080 m`
- Seeds: `[71001, 71002, 71003]`
- Official outcome gate: `mean_cable_progress_gap >= 0.01 m`
- Outcome decomposition is diagnostic only: `True`
- Motion stages successful: `True`
- Failed checks: `['preload_visibility', 'main_ade', 'main_fde', 'branch_amplification', 'vision_screen', 'sensor_over_vision_margin', 'outcome_progress_gap', 'fde_seed_fraction', 'amplification_seed_fraction']`
- Verdict: `HIDDEN_WIDE_STOP_SMOKE_BLOCKED`
- Training performed: `False`
- Geometry search performed: `False`
- Action search performed: `False`
- Status: fixed 3-seed Stage M0 smoke, not Scientific PASS.
