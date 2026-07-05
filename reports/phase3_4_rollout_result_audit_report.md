# Phase3.4 Rollout Result Audit

## Verdict

- Verdict: `WARN`
- Rows: `32`
- Primary pair: `free_vs_hidden_breakaway_pin`
- Diagnostic pair: `free_vs_hidden_pin`

## Success Table

| Baseline | Condition | Success | Total | Success rate | Final fraction mean | Runtime errors |
|---|---|---:|---:|---:|---:|---:|
| `paper_state` | `free` | 0 | 4 | 0.000 | 0.000 | 0 |
| `paper_state` | `hidden_breakaway_pin` | 0 | 4 | 0.000 | 0.000 | 0 |
| `paper_state` | `hidden_high_friction` | 0 | 4 | 0.000 | 0.073 | 0 |
| `paper_state` | `hidden_pin` | 0 | 4 | 0.000 | 0.000 | 0 |
| `state_action` | `free` | 0 | 4 | 0.000 | 0.052 | 0 |
| `state_action` | `hidden_breakaway_pin` | 0 | 4 | 0.000 | 0.000 | 0 |
| `state_action` | `hidden_high_friction` | 0 | 4 | 0.000 | 0.000 | 0 |
| `state_action` | `hidden_pin` | 0 | 4 | 0.000 | 0.000 | 0 |

## Success Gaps

| Baseline | Primary gap free-hidden_breakaway | Diagnostic gap free-hidden_pin |
|---|---:|---:|
| `paper_state` | 0.000 | 0.000 |
| `state_action` | 0.000 | 0.000 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `no_policy_success_observed` | All rollout smoke trials executed without runtime errors, but none reached task success. Treat this as runtime smoke only, not valid policy execution evidence. |

## Scope

- This is rollout smoke only.
- No Phase4 was run.
- No CPS was run.
- Do not make paper-level claims from this smoke.
