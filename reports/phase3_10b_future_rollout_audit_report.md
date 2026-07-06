# Phase3.10b Learned Future Quality + Rollout Error Audit Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase310b_predicted_future_branch_mismatch_supported`
- Episode rows: `36`
- Step rows: `576`
- Progress status: `completed`

## Per-Condition Diagnosis

| Condition | Old final | Default final | Best final | Default Δ | Best Δ | Old future match | Default future match | Best future match | Old pull | Default pull | Best pull |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free` | 0.1979 | 0.2500 | 0.0000 | 0.0521 | -0.1979 | 0.016 | 0.031 | 0.016 | 0.0605 | 0.0501 | 0.1494 |
| `hidden_breakaway_pin` | 0.0000 | 0.0000 | 0.0729 | 0.0000 | 0.0729 | 0.250 | 0.312 | 0.312 | 0.2831 | 0.0553 | 0.0677 |
| `hidden_high_friction` | 0.0417 | 0.0938 | 0.0312 | 0.0521 | -0.0104 | 0.703 | 0.719 | 0.672 | 0.1846 | 0.1622 | 0.0717 |

## Episode Summary

| Policy | Condition | Rows | OK | Timeout | Success | Final fraction | Future match | Future NN L2 | Input NN L2 | Chosen pull | Action OOD | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `old_state_action` | `free` | 4 | 4 | 0 | 0.000 | 0.1979 | 0.016 | 0.0453 | 0.0512 | 0.0605 | 0.230 | 0 |
| `old_state_action` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0.000 | 0.0000 | 0.250 | 0.0471 | 0.0864 | 0.2831 | 0.613 | 0 |
| `old_state_action` | `hidden_high_friction` | 4 | 4 | 0 | 0.000 | 0.0417 | 0.703 | 0.0558 | 0.0752 | 0.1846 | 0.529 | 0 |
| `phase39_default_geometry` | `free` | 4 | 4 | 0 | 0.000 | 0.2500 | 0.031 | 0.0451 | 0.0461 | 0.0501 | 0.252 | 0 |
| `phase39_default_geometry` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0.000 | 0.0000 | 0.312 | 0.0493 | 0.0476 | 0.0553 | 0.233 | 0 |
| `phase39_default_geometry` | `hidden_high_friction` | 4 | 4 | 0 | 0.000 | 0.0938 | 0.719 | 0.0559 | 0.0761 | 0.1622 | 0.435 | 0 |
| `phase39b_xy_only_high_weight` | `free` | 4 | 4 | 0 | 0.000 | 0.0000 | 0.016 | 0.0460 | 0.0738 | 0.1494 | 0.412 | 0 |
| `phase39b_xy_only_high_weight` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0.000 | 0.0729 | 0.312 | 0.0524 | 0.0502 | 0.0677 | 0.221 | 0 |
| `phase39b_xy_only_high_weight` | `hidden_high_friction` | 4 | 4 | 0 | 0.000 | 0.0312 | 0.672 | 0.0536 | 0.0462 | 0.0717 | 0.263 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `predicted_future_branch_mismatch_supported` | {'hidden_best_future_match': 0.3125, 'hidden_default_future_match': 0.3125, 'hidden_old_future_match': 0.25} |
| `WARN` | `predicted_future_action_pull_collapse_supported` | {'hidden_old_pull': 0.2830500003838097, 'hidden_default_pull': 0.05530500024178764, 'hidden_best_pull': 0.06771948139066808} |

## Interpretation

- This is a learned-future rollout error audit only.
- It uses offline y_state / condition_name only for diagnostic nearest-neighbor analysis, not model input.
- No model training was run.
- No future DDPM was trained.
- No Phase4 or CPS was run.
- If predicted future branch mismatch is supported, next step is future DDPM / conditioning diagnosis, not IDM repair or CPS.
