# Phase3 Input Consistency and Leakage Report

## Pair Consistency

| Metric | Value |
|---|---:|
| `num_pairs` | `7` |
| `mean_pair_paper_x_max_abs_diff` | `0.0` |
| `max_pair_paper_x_max_abs_diff` | `0.0` |
| `mean_pair_state_action_x_max_abs_diff` | `0.0` |
| `max_pair_state_action_x_max_abs_diff` | `0.0` |

## Robot Pose Proxy Sources

| Source | Count |
|---|---:|
| `pybullet_robot_body` | 1096 |

## Probe Accuracies

| Probe | Accuracy |
|---|---:|
| `paper_x -> condition_id` | `0.333333` |
| `state_action_x -> condition_id` | `0.333333` |
| `paper_x -> free_vs_pin` | `0.500000` |
| `paper_x -> success` | `0.571429` |

## Conclusion

PASS: paired `free` and `hidden_pin` inputs are consistent and probes do not recover hidden condition above the failure threshold.
