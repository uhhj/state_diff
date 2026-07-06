# Phase3.9 Geometry-Aware IDM Train Report

## Verdict

- Verdict: `PASS`
- Baseline: `state_action`
- Future key: `y_state`
- Train rows: `768`
- Val rows: `188`
- Inverse dynamics path: `/data/state_diff2/checkpoints/phase3_9b_geometry_idm/state_action/no_coupled_xy_loss/fold_phase3_9b_seed_392002/inverse_dynamics.pt`

## Validation Metrics

| Metric | Value |
|---|---:|
| `val_action_mae` | `0.008239` |
| `val_pose0_xy_mae` | `0.027626` |
| `val_pose1_xy_mae` | `0.026053` |
| `val_pull_xy_mae` | `0.036390` |

## Scope

- Inverse dynamics only.
- No future DDPM training.
- No Phase4.
- No CPS.
- Checkpoint must not be committed.
