# Phase3 Condition Integration Completion Audit

## Verdict

- Verdict: `PASS`
- Mode completed: `medium`
- Conditions: `free hidden_pin hidden_high_friction hidden_breakaway_pin`
- Primary branch pair: `free_vs_hidden_breakaway_pin`
- Diagnostic branch pair: `free_vs_hidden_pin`
- Selected Phase2.5c config: `breakaway_force_2p6_disp_0p045_pull_0p36`

## Data And Windows

| Field | Value |
|---|---:|
| train visible seeds | 100 |
| heldout visible seeds | 30 |
| windows | 956 |
| prediction rows | 2256 |

## Condition Windows

| Condition | Windows |
|---|---:|
| `free` | 239 |
| `hidden_breakaway_pin` | 239 |
| `hidden_high_friction` | 239 |
| `hidden_pin` | 239 |

## Guards

| Guard | Result |
|---|---|
| y_action shape | `[956, 14]` |
| action dim | `14` |
| camera_config paths in action codec | `0` |
| state_action extra std | `0.2338666021823883` |
| input leakage pass | `True` |
| max post-canonical pair diff | `0.0` |
| backend counts | `{'torch': 2256}` |
| missing primary branch refs | `0` |
| sanity verdict | `WARN` |
| action/IDM verdict | `WARN` |
| audit verdict | `WARN` |

## Scope Limits

- Policy rollout was not run.
- Phase4/CPS was not started.
- `hidden_pin` remains a hard diagnostic branch; `hidden_breakaway_pin` is the primary recoverable branch for this Phase3 medium run.

## Issues

- No FAIL issues recorded. WARN items are documented in the individual reports.
