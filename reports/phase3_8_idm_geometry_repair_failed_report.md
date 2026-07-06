# Phase3.8 IDM Geometry Repair Probe Report

## Verdict

- Verdict: `FAIL`
- Reason: `gated_repair_probe_did_not_complete`
- The repair probe was explicitly gated and launched, but it ran for more than two hours without producing `phase3_8_idm_geometry_repair_trials.csv` or raw summary JSON.
- The long-running process was terminated to avoid leaving residual PyBullet execution processes.

## What Completed

- Preflight verdict: `PASS`
- Action geometry sensitivity verdict: `PASS`
- Decode/action sensitivity completed and points to pose geometry, especially pose1 XY, as the likely blocker.

## What Did Not Complete

- Gated repair execution trials did not complete.
- No deployable rollout or CPS evidence was generated.
- GT-blended repair variants remain diagnostic upper bounds only.

## Top Sensitivity Dimensions

| Rank | Baseline | Source | Dim | Path | Component | Mean abs | Max abs | Mean signed |
|---:|---|---|---:|---|---|---:|---:|---:|
| 1 | `paper_state` | `idm_pred_future` | 8 | `params/pose1/0/1` | `pose1_xyz` | 0.145582 | 0.283119 | -0.140667 |
| 2 | `state_action` | `idm_pred_future` | 8 | `params/pose1/0/1` | `pose1_xyz` | 0.126913 | 0.245972 | -0.110699 |
| 3 | `paper_state` | `idm_pred_future` | 1 | `params/pose0/0/1` | `pose0_xyz` | 0.095614 | 0.348677 | -0.062940 |
| 4 | `state_action` | `idm_pred_future` | 1 | `params/pose0/0/1` | `pose0_xyz` | 0.087544 | 0.351000 | -0.042467 |
| 5 | `paper_state` | `idm_pred_future` | 7 | `params/pose1/0/0` | `pose1_xyz` | 0.073225 | 0.169461 | 0.059522 |
| 6 | `state_action` | `idm_pred_future` | 7 | `params/pose1/0/0` | `pose1_xyz` | 0.070984 | 0.191543 | 0.053303 |
| 7 | `paper_state` | `idm_gt_future` | 8 | `params/pose1/0/1` | `pose1_xyz` | 0.061483 | 0.169924 | -0.052633 |
| 8 | `state_action` | `idm_gt_future` | 8 | `params/pose1/0/1` | `pose1_xyz` | 0.061483 | 0.169924 | -0.052633 |

## Interpretation

- Phase3.7 root cause remains refined toward IDM action execution geometry: pose0/pose1 XY geometry, especially pose1 XY, is the dominant suspect.
- Because the repair execution batch did not finish, Phase3.8 does not clear the blocker.
- Do not enter Phase4 or CPS from this state.

## Next Step

- Add incremental CSV flushing / per-row progress and a bounded repair matrix, or rerun with explicit user-approved narrower settings.
- Any narrowed run must be reported as such and cannot be confused with the full Phase3.8 repair matrix.
