# Phase3.4 Rollout Smoke Report

## Scope

- This is small rollout smoke only.
- It is not Phase4 and not CPS evidence.
- Primary pair: `free_vs_hidden_breakaway_pin`
- Diagnostic pair: `free_vs_hidden_pin`
- `free_vs_hidden_pin` is diagnostic only.

## Summary

| Baseline | Condition | Success | Total | Rate | Final fraction mean | Action OOD mean |
|---|---|---:|---:|---:|---:|---:|
| `paper_state` | `free` | 0 | 4 | 0.000 | 0.000 | 0.253 |
| `paper_state` | `hidden_breakaway_pin` | 0 | 4 | 0.000 | 0.000 | 0.220 |
| `paper_state` | `hidden_high_friction` | 0 | 4 | 0.000 | 0.073 | 0.240 |
| `paper_state` | `hidden_pin` | 0 | 4 | 0.000 | 0.000 | 0.422 |
| `state_action` | `free` | 0 | 4 | 0.000 | 0.052 | 0.193 |
| `state_action` | `hidden_breakaway_pin` | 0 | 4 | 0.000 | 0.000 | 0.237 |
| `state_action` | `hidden_high_friction` | 0 | 4 | 0.000 | 0.000 | 0.249 |
| `state_action` | `hidden_pin` | 0 | 4 | 0.000 | 0.000 | 0.447 |

## Success Gaps

| Baseline | Primary success gap free-hidden_breakaway | Diagnostic gap free-hidden_pin |
|---|---:|---:|
| `paper_state` | 0.000 | 0.000 |
| `state_action` | 0.000 | 0.000 |
