# Phase1 Hidden-Contact Cable Report

## Summary

- Task: `hidden-contact-cable-line`
- Data root: `/data/state_diff2/external/deformable-ravens/data/hidden-contact-cable-line`
- Conditions: `free, hidden_pin, hidden_high_friction`

## Condition Summary

| Condition | Episodes | Success Rate | Mean Final Fraction | Mean Final Curve |
|---|---:|---:|---:|---:|
| free | 5 | 1.0000 | 1.0000 | 0.0028 |
| hidden_pin | 5 | 0.0000 | 0.0000 | 0.0127 |
| hidden_high_friction | 5 | 1.0000 | 1.0000 | 0.0036 |

## Paired Free-vs-Hidden Metrics

| Seed | Pair | Initial Chamfer | Final Chamfer | Success Diff |
|---|---|---:|---:|---:|
| 0 | free vs hidden_high_friction | 0.0003 | 0.0136 | False |
| 0 | free vs hidden_pin | 0.0001 | 0.1965 | True |
| 1 | free vs hidden_high_friction | 0.0001 | 0.0154 | False |
| 1 | free vs hidden_pin | 0.0001 | 0.7048 | True |
| 2 | free vs hidden_high_friction | 0.0004 | 0.0184 | False |
| 2 | free vs hidden_pin | 0.0001 | 0.2509 | True |
| 3 | free vs hidden_high_friction | 0.0003 | 0.0137 | False |
| 3 | free vs hidden_pin | 0.0001 | 0.1442 | True |
| 4 | free vs hidden_high_friction | 0.0005 | 0.0160 | False |
| 4 | free vs hidden_pin | 0.0001 | 0.3797 | True |

## Phase1 Interpretation

- This phase verifies task registration, hidden-contact injection, bead-state logging, and qualitative/quantitative divergence.
- It is not yet the final CCDA audit. Phase2 should generate larger paired rollouts and compute formal thresholds.
- A good Phase1 signal is low initial Chamfer with higher final Chamfer between `free` and hidden-contact conditions.

## Phase1 Cleanup Conclusion

Phase1 now keeps only two hidden-contact variants: `hidden_pin` and `hidden_high_friction`.
The invalid side-jam variant was removed because RGB-D observation checks showed visible leakage.
The previous action-step observation replay visualization was also removed because it did not show continuous robot-cable contact and could be misleading.
The retained visual checks are bead trajectory overlays and RGB-D observation checks.
Continuous robot-cable interaction should be inspected only with the continuous PyBullet rollout recorder.
