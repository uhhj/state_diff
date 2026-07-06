# Phase3.5 Action Execution Diagnostic Report

## Verdict

- Verdict: `FAIL`
- Root cause: `action_codec_or_rollout_adapter_blocker`
- Rows: `246`
- Oracle success count: `8`
- GT y_action replay success count: `0`
- Learned action success count: `0`

## Summary Table

| Source | Baseline | Condition | Timeout | Rows | Success | Rate | Failures | Valid action rate | Δ final fraction | Final fraction | OOD | Clip fraction |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gt_y_action_replay` | `` | `free` | 15.0 | 2 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | nan | nan |
| `gt_y_action_replay` | `` | `free` | 5.0 | 2 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | nan | nan |
| `gt_y_action_replay` | `` | `hidden_breakaway_pin` | 15.0 | 2 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | nan | nan |
| `gt_y_action_replay` | `` | `hidden_breakaway_pin` | 5.0 | 2 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | nan | nan |
| `gt_y_action_replay` | `` | `hidden_high_friction` | 15.0 | 2 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | nan | nan |
| `gt_y_action_replay` | `` | `hidden_high_friction` | 5.0 | 2 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | nan | nan |
| `gt_y_action_replay` | `` | `hidden_pin` | 15.0 | 2 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | nan | nan |
| `gt_y_action_replay` | `` | `hidden_pin` | 5.0 | 2 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | nan | nan |
| `learned_action` | `paper_state` | `free` | 15.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.016 | 0.062 | 0.396 | 0.000 |
| `learned_action` | `paper_state` | `free` | 5.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.047 | 0.413 | 0.000 |
| `learned_action` | `paper_state` | `hidden_breakaway_pin` | 15.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.382 | 0.000 |
| `learned_action` | `paper_state` | `hidden_breakaway_pin` | 5.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.010 | 0.042 | 0.394 | 0.000 |
| `learned_action` | `paper_state` | `hidden_high_friction` | 15.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.584 | 0.000 |
| `learned_action` | `paper_state` | `hidden_high_friction` | 5.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.031 | 0.403 | 0.000 |
| `learned_action` | `paper_state` | `hidden_pin` | 15.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.634 | 0.000 |
| `learned_action` | `paper_state` | `hidden_pin` | 5.0 | 5 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.543 | 0.000 |
| `learned_action` | `state_action` | `free` | 15.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.016 | 0.308 | 0.000 |
| `learned_action` | `state_action` | `free` | 5.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.036 | 0.308 | 0.000 |
| `learned_action` | `state_action` | `hidden_breakaway_pin` | 15.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.026 | 0.068 | 0.385 | 0.000 |
| `learned_action` | `state_action` | `hidden_breakaway_pin` | 5.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.355 | 0.000 |
| `learned_action` | `state_action` | `hidden_high_friction` | 15.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.546 | 0.000 |
| `learned_action` | `state_action` | `hidden_high_friction` | 5.0 | 8 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.585 | 0.000 |
| `learned_action` | `state_action` | `hidden_pin` | 15.0 | 2 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.573 | 0.000 |
| `learned_action` | `state_action` | `hidden_pin` | 5.0 | 5 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.701 | 0.000 |
| `oracle_action` | `` | `free` | 15.0 | 16 | 2 | 0.125 | 0 | 0.875 | 0.125 | 0.555 | nan | nan |
| `oracle_action` | `` | `free` | 5.0 | 16 | 2 | 0.125 | 0 | 0.875 | 0.125 | 0.495 | nan | nan |
| `oracle_action` | `` | `hidden_breakaway_pin` | 15.0 | 16 | 2 | 0.125 | 0 | 0.875 | 0.125 | 0.492 | nan | nan |
| `oracle_action` | `` | `hidden_breakaway_pin` | 5.0 | 16 | 0 | 0.000 | 0 | 0.875 | 0.112 | 0.461 | nan | nan |
| `oracle_action` | `` | `hidden_high_friction` | 15.0 | 15 | 1 | 0.067 | 0 | 0.867 | 0.131 | 0.456 | nan | nan |
| `oracle_action` | `` | `hidden_high_friction` | 5.0 | 14 | 1 | 0.071 | 0 | 0.857 | 0.137 | 0.432 | nan | nan |
| `oracle_action` | `` | `hidden_pin` | 15.0 | 11 | 0 | 0.000 | 0 | 0.818 | 0.000 | 0.000 | nan | nan |
| `oracle_action` | `` | `hidden_pin` | 5.0 | 10 | 0 | 0.000 | 0 | 0.800 | 0.000 | 0.000 | nan | nan |

## Timeout Summary

| Timeout | Rows | Success rate | Failure count | Mean Δ final fraction |
|---:|---:|---:|---:|---:|
| 5.0 | 122 | 0.025 | 0 | 0.047 |
| 15.0 | 124 | 0.040 | 0 | 0.051 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `FAIL` | `gt_y_action_replay_no_progress_but_oracle_progresses` | Oracle progresses but decoded y_action replay does not. Suspect action codec / coordinate scale / action-template mismatch. |
| `WARN` | `motion_timeout_affects_success` | success_rate improves from timeout 5.0 to 15.0; Phase3.4 may have been timeout-limited. |

## Interpretation

- This is action execution diagnostic only.
- It is not Phase4 and not CPS evidence.
- Oracle/GT failures point to environment/adapter/action-codec issues.
- Oracle/GT progress with learned failure points to learned policy or closed-loop alignment.
