# Phase3 Condition Integration Audit

## Verdict

- Verdict: `PASS`
- Conditions: `free, hidden_pin, hidden_high_friction, hidden_breakaway_pin`
- Primary branch pair: `free_vs_hidden_breakaway_pin`
- Diagnostic branch pair: `free_vs_hidden_pin`
- Phase2.5 selected config: `breakaway_force_2p6_disp_0p045_pull_0p36`

## Window Balance

| Condition | Windows |
|---|---:|
| `free` | 239 |
| `hidden_pin` | 239 |
| `hidden_high_friction` | 239 |
| `hidden_breakaway_pin` | 239 |

## Leakage Guard

- Input leakage pass: `True`
- Max post-canonical pair diff: `0.0`
- Action dim: `14`
- y_action shape: `[956, 14]`
- State-action extra std: `0.2338666021823883`

## Issues

| Level | Name | Detail |
|---|---|---|
| `PASS` | `none` | No blocking issue found. |

## Conclusion

PASS: Phase3 is configured for the Phase2.5c recoverable branch without exposing hidden-contact labels, branch names, success labels, or recoverability parameters to model inputs.
