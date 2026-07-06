# Phase3.9b Geometry IDM Robustness + Loss Ablation Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase39b_geometry_loss_ablation_supported`
- Best ablation: `xy_only_high_weight`
- Default robust: `True`
- Coupled loss needed: `False`
- Pull loss needed: `True`
- Quat loss not critical: `False`
- XY-only high-weight competitive: `True`

## Ablation Summary

| Ablation | Verdict | Root cause | Improved conditions | Primary improved | Hidden breakaway improvement | Mean improvement | Mean Phase3.9 Δ | Mean old Δ |
|---|---|---|---|---:|---:|---:|---:|---:|
| `default_geometry` | `WARN` | `phase39_geometry_idm_repair_supported` | `['free', 'hidden_breakaway_pin', 'hidden_high_friction']` | `True` | 0.0694 | 0.1620 | 0.2083 | 0.0463 |
| `no_pull_loss` | `FAIL` | `phase39_repair_not_supported` | `['hidden_high_friction']` | `False` | -0.0694 | 0.1227 | 0.1597 | 0.0370 |
| `no_coupled_xy_loss` | `WARN` | `phase39_geometry_idm_repair_supported` | `['free', 'hidden_breakaway_pin', 'hidden_high_friction']` | `True` | 0.1944 | 0.3449 | 0.3704 | 0.0255 |
| `xy_only_high_weight` | `WARN` | `phase39_geometry_idm_repair_supported` | `['free', 'hidden_breakaway_pin', 'hidden_high_friction']` | `True` | 0.2361 | 0.3148 | 0.3287 | 0.0139 |
| `no_quat_loss` | `WARN` | `phase39_geometry_idm_repair_supported` | `['free', 'hidden_breakaway_pin', 'hidden_high_friction']` | `True` | 0.2222 | 0.4444 | 0.4861 | 0.0417 |

## Per-Ablation Per-Condition

| Ablation | Condition | GT Δ | Old IDM Δ | New IDM Δ | Improvement | Improved |
|---|---|---:|---:|---:|---:|---:|
| `default_geometry` | `free` | 0.2917 | 0.0000 | 0.2500 | 0.2500 | `True` |
| `default_geometry` | `hidden_breakaway_pin` | 0.2361 | 0.0556 | 0.1250 | 0.0694 | `True` |
| `default_geometry` | `hidden_high_friction` | 0.1875 | 0.0833 | 0.2500 | 0.1667 | `True` |
| `no_pull_loss` | `free` | 0.2917 | 0.0000 | 0.0000 | 0.0000 | `False` |
| `no_pull_loss` | `hidden_breakaway_pin` | 0.1806 | 0.0694 | 0.0000 | -0.0694 | `False` |
| `no_pull_loss` | `hidden_high_friction` | 0.7083 | 0.0417 | 0.4792 | 0.4375 | `True` |
| `no_coupled_xy_loss` | `free` | 0.2778 | 0.0000 | 0.2361 | 0.2361 | `True` |
| `no_coupled_xy_loss` | `hidden_breakaway_pin` | 0.2083 | 0.0139 | 0.2083 | 0.1944 | `True` |
| `no_coupled_xy_loss` | `hidden_high_friction` | 0.4583 | 0.0625 | 0.6667 | 0.6042 | `True` |
| `xy_only_high_weight` | `free` | 0.2778 | 0.0000 | 0.2917 | 0.2917 | `True` |
| `xy_only_high_weight` | `hidden_breakaway_pin` | 0.1944 | 0.0417 | 0.2778 | 0.2361 | `True` |
| `xy_only_high_weight` | `hidden_high_friction` | 0.9167 | 0.0000 | 0.4167 | 0.4167 | `True` |
| `no_quat_loss` | `free` | 0.2917 | 0.0000 | 0.1944 | 0.1944 | `True` |
| `no_quat_loss` | `hidden_breakaway_pin` | 0.2222 | 0.0417 | 0.2639 | 0.2222 | `True` |
| `no_quat_loss` | `hidden_high_friction` | 0.8750 | 0.0833 | 1.0000 | 0.9167 | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `default_geometry_repair_robust` | {'improved_conditions': ['free', 'hidden_breakaway_pin', 'hidden_high_friction'], 'hidden_breakaway_improvement': 0.06944444444444445} |
| `WARN` | `pull_xy_loss_supported_by_ablation` | hidden_breakaway improvement gap=0.138889 |
| `WARN` | `xy_only_high_weight_competitive` | hidden_breakaway improvement gap=0.166667 |

## Interpretation

- This is one-step matched-prefix diagnostic only.
- No Phase4 or CPS was run.
- No future DDPM was trained.
- Checkpoints are local diagnostic artifacts and must not be committed.
- If default geometry is robust, the next step is Phase3.10 controlled learned rollout retry, not Phase4/CPS.
