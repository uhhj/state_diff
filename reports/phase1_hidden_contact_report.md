# Phase1 Hidden-Contact Cable Report

## Summary

- Task: `hidden-contact-cable-line`
- Data root: `/data/state_diff2/external/deformable-ravens/data/hidden-contact-cable-line`
- Conditions: `free, hidden_pin, hidden_high_friction, hidden_side_jam`

## Condition Summary

| Condition | Episodes | Success Rate | Mean Final Fraction | Mean Final Curve |
|---|---:|---:|---:|---:|
| free | 5 | 1.0000 | 1.0000 | 0.0028 |
| hidden_pin | 5 | 0.0000 | 0.0000 | 0.0129 |
| hidden_high_friction | 5 | 1.0000 | 1.0000 | 0.0050 |
| hidden_side_jam | 5 | 1.0000 | 1.0000 | 0.0025 |

## Paired Free-vs-Hidden Metrics

| Seed | Pair | Initial Chamfer | Final Chamfer | Success Diff |
|---|---|---:|---:|---:|
| 0 | free vs hidden_high_friction | 0.0003 | 0.0230 | False |
| 0 | free vs hidden_pin | 0.0001 | 0.1949 | True |
| 0 | free vs hidden_side_jam | 0.0108 | 0.0138 | False |
| 1 | free vs hidden_high_friction | 0.0002 | 0.0143 | False |
| 1 | free vs hidden_pin | 0.0001 | 0.6896 | True |
| 1 | free vs hidden_side_jam | 0.0111 | 0.0239 | False |
| 2 | free vs hidden_high_friction | 0.0005 | 0.0179 | False |
| 2 | free vs hidden_pin | 0.0000 | 0.2439 | True |
| 2 | free vs hidden_side_jam | 0.0075 | 0.0151 | False |
| 3 | free vs hidden_high_friction | 0.0003 | 0.0315 | False |
| 3 | free vs hidden_pin | 0.0002 | 0.1508 | True |
| 3 | free vs hidden_side_jam | 0.0119 | 0.0126 | False |
| 4 | free vs hidden_high_friction | 0.0005 | 0.0195 | False |
| 4 | free vs hidden_pin | 0.0001 | 0.3999 | True |
| 4 | free vs hidden_side_jam | 0.0094 | 0.0140 | False |

## Phase1 Interpretation

- This phase verifies task registration, hidden-contact injection, bead-state logging, and qualitative/quantitative divergence.
- It is not yet the final CCDA audit. Phase2 should generate larger paired rollouts and compute formal thresholds.
- A good Phase1 signal is low initial Chamfer with higher final Chamfer between `free` and hidden-contact conditions.
