# Phase2 CCDA Audit Report

## Summary

- Task: `hidden-contact-cable-line`
- Data root: `/data/state_diff2/external/deformable-ravens/data/phase2_hidden_contact_cable_line/hidden-contact-cable-line`
- Conditions: `free, hidden_pin, hidden_high_friction`
- Total paired comparisons: `150`
- CCDA-state count: `97`
- CCDA-state ratio: `0.6467`
- CCDA-success count: `97`
- CCDA-success ratio: `0.6467`
- CCDA impact gap: `0.9434`

## Thresholds

| Metric | Threshold | Meaning |
|---|---:|---|
| `tau_vis_initial_chamfer` | `0.002` | |
| `tau_action_first_action` | `0.02` | |
| `tau_future_final_chamfer` | `0.05` | |
| `tau_rgb_mean_abs_diff` | `1.0` | |
| `tau_depth_mean_abs_diff` | `0.0001` | |

## Condition Summary

| Condition | Episodes | Success Rate | Mean Final Fraction | Mean Final Curve |
|---|---:|---:|---:|---:|
| free | 50 | 1.0000 | 1.0000 | 0.0037 |
| hidden_pin | 50 | 0.0000 | 0.0467 | 0.0118 |
| hidden_high_friction | 50 | 1.0000 | 1.0000 | 0.0044 |

## Pair-Type Summary

| Pair | Count | CCDA-State Ratio | CCDA-Success Ratio | SuccessDiff Ratio | Init Chamfer | Action Dist | Final Chamfer | RGB Mean Diff | Depth Mean Diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| free_vs_hidden_pin | 50 | 0.9800 | 0.9800 | 1.0000 | 0.0002 | 0.0000 | 0.4345 | 0.0016 | 0.0000 |
| free_vs_hidden_high_friction | 50 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 0.0000 | 0.0172 | 0.0044 | 0.0000 |
| hidden_pin_vs_hidden_high_friction | 50 | 0.9600 | 0.9600 | 1.0000 | 0.0004 | 0.0000 | 0.4354 | 0.0041 | 0.0000 |

## RGB-D Leakage Check

| Pair | Count | Mean RGB Abs Diff | Max RGB Abs Diff | Mean Depth Abs Diff | Max Depth Abs Diff |
|---|---:|---:|---:|---:|---:|
| free_vs_hidden_pin | 50 | 0.0016 | 184.0000 | 0.0000 | 0.0372 |
| free_vs_hidden_high_friction | 50 | 0.0044 | 193.0000 | 0.0000 | 0.0447 |
| hidden_pin_vs_hidden_high_friction | 50 | 0.0041 | 193.0000 | 0.0000 | 0.0447 |

## Failure Reason Counts

| Reason | Count |
|---|---:|
| `ccda_state` | 97 |
| `future_not_different` | 51 |
| `visible_not_similar_or_rgbd_leakage` | 2 |

## Branch Pair Counts

| Branch Pair | Count |
|---|---:|
| `pinned_bend vs weak_contact_success` | 50 |
| `success_line vs pinned_bend` | 50 |
| `success_line vs weak_contact_success` | 50 |

## Top CCDA-Success Examples

| Seed | Pair | Init Chamfer | Action Dist | Final Chamfer | Success A | Success B | Branch A | Branch B |
|---|---|---:|---:|---:|---:|---:|---|---|
| 37 | hidden_pin vs hidden_high_friction | 0.0003 | 0.0000 | 0.9458 | False | True | pinned_bend | weak_contact_success |
| 37 | free vs hidden_pin | 0.0001 | 0.0000 | 0.9355 | True | False | success_line | pinned_bend |
| 15 | free vs hidden_pin | 0.0001 | 0.0000 | 0.9100 | True | False | success_line | pinned_bend |
| 29 | free vs hidden_pin | 0.0001 | 0.0000 | 0.9074 | True | False | success_line | pinned_bend |
| 15 | hidden_pin vs hidden_high_friction | 0.0002 | 0.0000 | 0.9014 | False | True | pinned_bend | weak_contact_success |
| 38 | free vs hidden_pin | 0.0002 | 0.0000 | 0.8978 | True | False | success_line | pinned_bend |
| 38 | hidden_pin vs hidden_high_friction | 0.0002 | 0.0000 | 0.8946 | False | True | pinned_bend | weak_contact_success |
| 28 | hidden_pin vs hidden_high_friction | 0.0001 | 0.0000 | 0.8852 | False | True | pinned_bend | weak_contact_success |
| 28 | free vs hidden_pin | 0.0000 | 0.0000 | 0.8849 | True | False | success_line | pinned_bend |
| 29 | hidden_pin vs hidden_high_friction | 0.0004 | 0.0000 | 0.8607 | False | True | pinned_bend | weak_contact_success |
| 47 | free vs hidden_pin | 0.0001 | 0.0000 | 0.7730 | True | False | success_line | pinned_bend |
| 47 | hidden_pin vs hidden_high_friction | 0.0004 | 0.0000 | 0.7717 | False | True | pinned_bend | weak_contact_success |
| 21 | free vs hidden_pin | 0.0005 | 0.0000 | 0.7524 | True | False | success_line | pinned_bend |
| 31 | hidden_pin vs hidden_high_friction | 0.0001 | 0.0000 | 0.7428 | False | True | pinned_bend | weak_contact_success |
| 45 | hidden_pin vs hidden_high_friction | 0.0003 | 0.0001 | 0.7399 | False | True | pinned_bend | weak_contact_success |
| 20 | hidden_pin vs hidden_high_friction | 0.0004 | 0.0000 | 0.7391 | False | True | pinned_bend | weak_contact_success |
| 45 | free vs hidden_pin | 0.0001 | 0.0000 | 0.7384 | True | False | success_line | pinned_bend |
| 21 | hidden_pin vs hidden_high_friction | 0.0006 | 0.0000 | 0.7359 | False | True | pinned_bend | weak_contact_success |
| 31 | free vs hidden_pin | 0.0001 | 0.0000 | 0.7230 | True | False | success_line | pinned_bend |
| 49 | free vs hidden_pin | 0.0001 | 0.0000 | 0.7228 | True | False | success_line | pinned_bend |

## Phase2 Conclusion

Phase2 confirms a formal CCDA dataset for `hidden-contact-cable-line`: the `free` and `hidden_pin` paired rollouts keep matched visible initial state, matched first action, and RGB-D observations below the leakage thresholds, while their future cable states and success outcomes diverge.

Phase2 keeps only two hidden-contact variants: `hidden_pin` and `hidden_high_friction`. `hidden_side_jam` remains removed because RGB-D observation checks showed visible leakage. The previous action-step observation replay visualization remains removed because it did not show continuous robot-cable contact and could be misleading. The retained visual checks are bead trajectory overlays and RGB-D observation checks. Continuous robot-cable interaction should be inspected only with the continuous PyBullet rollout recorder.

## Interpretation

- `CCDA-state` means visible state, RGB-D observation, and first action are similar, contact condition differs, and future cable state diverges.
- `CCDA-success` additionally requires different success outcomes.
- `hidden_pin` is expected to be the strong CCDA branch.
- `hidden_high_friction` is retained as a weak-contact perturbation; it may show small future divergence without success difference.
- If RGB-D differences are large, the hidden-contact premise is violated and the condition should not be used as CCDA evidence.
