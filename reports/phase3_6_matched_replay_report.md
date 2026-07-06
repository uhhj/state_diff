# Phase3.6 Matched-State Action Replay Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase35_unmatched_replay_artifact_possible`
- Rows: `24`
- Prefix sources: `['raw_action_file_high_confidence']`
- state_match: `True`
- mean_prefix_state_mae: `0.055298`
- GT success count: `2`
- Oracle success count: `1`
- GT mean Δ final fraction: `0.204167`
- Oracle mean Δ final fraction: `0.117424`

## Summary Table

| Test | Condition | Prefix source | Timeout | Rows | Success | Rate | Failures | Valid action | Δ final fraction | Final fraction | Prefix MAE | Prefix failure |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `matched_gt_y_action` | `free` | `raw_action_file_high_confidence` | 15.0 | 3 | 0 | 0.000 | 0 | 1.000 | 0.306 | 0.306 | 0.065 | 0.000 |
| `matched_gt_y_action` | `hidden_breakaway_pin` | `raw_action_file_high_confidence` | 15.0 | 3 | 0 | 0.000 | 0 | 1.000 | 0.208 | 0.208 | 0.065 | 0.000 |
| `matched_gt_y_action` | `hidden_high_friction` | `raw_action_file_high_confidence` | 15.0 | 3 | 2 | 0.667 | 0 | 1.000 | 0.500 | 0.500 | 0.033 | 0.000 |
| `matched_gt_y_action` | `hidden_pin` | `raw_action_file_high_confidence` | 15.0 | 3 | 0 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.060 | 0.000 |
| `same_state_oracle` | `free` | `raw_action_file_high_confidence` | 15.0 | 3 | 0 | 0.000 | 0 | 0.667 | 0.222 | 0.222 | 0.065 | 0.000 |
| `same_state_oracle` | `hidden_breakaway_pin` | `raw_action_file_high_confidence` | 15.0 | 3 | 0 | 0.000 | 0 | 0.667 | 0.236 | 0.236 | 0.065 | 0.000 |
| `same_state_oracle` | `hidden_high_friction` | `raw_action_file_high_confidence` | 15.0 | 3 | 1 | 0.333 | 0 | 0.333 | -0.042 | 0.417 | 0.028 | 0.000 |
| `same_state_oracle` | `hidden_pin` | `raw_action_file_high_confidence` | 15.0 | 3 | 0 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | 0.061 | 0.000 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `phase35_gt_failure_may_be_unmatched_reset_artifact` | Matched GT y_action shows progress; Phase3.5 unmatched replay failure should not be interpreted as codec failure. |

## Interpretation

- If prefix state mismatch is high, Phase3.6 cannot confirm codec/adapter failure.
- If prefix state matches and oracle progresses but matched GT does not, codec/adapter blocker is confirmed.
- If matched GT progresses, Phase3.5 GT failure was likely an unmatched-reset artifact.
- This is not Phase4 and not CPS evidence.
