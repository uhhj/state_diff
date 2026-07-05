# Phase3 Input Consistency and Leakage Report

## Condition Scope

- Conditions: `free, hidden_pin, hidden_high_friction, hidden_breakaway_pin`
- Primary pair: `free_vs_hidden_breakaway_pin`
- Diagnostic pair: `free_vs_hidden_pin`

## Pair Consistency By Hidden Condition

| Hidden condition | Pairs | Max paper_x diff | Max state_action_x diff |
|---|---:|---:|---:|
| `hidden_pin` | 47 | `0.0` | `0.0` |
| `hidden_high_friction` | 47 | `0.0` | `0.0` |
| `hidden_breakaway_pin` | 47 | `0.0` | `0.0` |

## Robot Pose Proxy Sources

| Source | Count |
|---|---:|
| `pybullet_robot_body` | 13471 |

## Probe Accuracies

| Probe | Accuracy |
|---|---:|
| `paper_x -> condition_id` | `0.250000` |
| `state_action_x -> condition_id` | `0.250000` |
| `paper_x -> free_vs_hidden_breakaway_pin` | `0.500000` |
| `paper_x -> free_vs_hidden_pin` | `0.500000` |
| `paper_x -> success` | `0.462766` |

## Forbidden Metadata Schema Check

- Pass: `True`
- Missing required forbidden entries: `[]`
- Forbidden names present in x schema: `[]`

## Conclusion

PASS: paired `free` and hidden-branch inputs are identical after canonicalization, and hidden/contact metadata remains outside model inputs.
