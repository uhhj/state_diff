# Phase3.8 Action Geometry Sensitivity

## Verdict

- Verdict: `PASS`
- Dimension rows: `672`
- Selected windows: `12`

## Component Error Table

| Baseline | Source | Component | Mean abs error | Max abs error | Mean signed error |
|---|---|---|---:|---:|---:|
| `paper_state` | `idm_gt_future` | `pose0_quat` | 0.000000 | 0.000000 | -0.000000 |
| `paper_state` | `idm_gt_future` | `pose0_xyz` | 0.026310 | 0.214663 | 0.018294 |
| `paper_state` | `idm_gt_future` | `pose1_quat` | 0.000000 | 0.000000 | -0.000000 |
| `paper_state` | `idm_gt_future` | `pose1_xyz` | 0.034315 | 0.169924 | -0.026399 |
| `paper_state` | `idm_pred_future` | `pose0_quat` | 0.000000 | 0.000000 | 0.000000 |
| `paper_state` | `idm_pred_future` | `pose0_xyz` | 0.049766 | 0.348677 | -0.008849 |
| `paper_state` | `idm_pred_future` | `pose1_quat` | 0.000000 | 0.000000 | -0.000000 |
| `paper_state` | `idm_pred_future` | `pose1_xyz` | 0.074575 | 0.283119 | -0.025665 |
| `state_action` | `idm_gt_future` | `pose0_quat` | 0.000000 | 0.000000 | -0.000000 |
| `state_action` | `idm_gt_future` | `pose0_xyz` | 0.026310 | 0.214663 | 0.018294 |
| `state_action` | `idm_gt_future` | `pose1_quat` | 0.000000 | 0.000000 | -0.000000 |
| `state_action` | `idm_gt_future` | `pose1_xyz` | 0.034315 | 0.169924 | -0.026399 |
| `state_action` | `idm_pred_future` | `pose0_quat` | 0.000000 | 0.000000 | -0.000000 |
| `state_action` | `idm_pred_future` | `pose0_xyz` | 0.048851 | 0.351000 | -0.001531 |
| `state_action` | `idm_pred_future` | `pose1_quat` | 0.000000 | 0.000000 | -0.000000 |
| `state_action` | `idm_pred_future` | `pose1_xyz` | 0.067475 | 0.245972 | -0.018058 |

## Top Error Dimensions

| Baseline | Source | Dim | Path | Component | Mean abs | Max abs | Mean signed |
|---|---|---:|---|---|---:|---:|---:|
| `paper_state` | `idm_pred_future` | 8 | `params/pose1/0/1` | `pose1_xyz` | 0.145582 | 0.283119 | -0.140667 |
| `state_action` | `idm_pred_future` | 8 | `params/pose1/0/1` | `pose1_xyz` | 0.126913 | 0.245972 | -0.110699 |
| `paper_state` | `idm_pred_future` | 1 | `params/pose0/0/1` | `pose0_xyz` | 0.095614 | 0.348677 | -0.062940 |
| `state_action` | `idm_pred_future` | 1 | `params/pose0/0/1` | `pose0_xyz` | 0.087544 | 0.351000 | -0.042467 |
| `paper_state` | `idm_pred_future` | 7 | `params/pose1/0/0` | `pose1_xyz` | 0.073225 | 0.169461 | 0.059522 |
| `state_action` | `idm_pred_future` | 7 | `params/pose1/0/0` | `pose1_xyz` | 0.070984 | 0.191543 | 0.053303 |
| `paper_state` | `idm_gt_future` | 8 | `params/pose1/0/1` | `pose1_xyz` | 0.061483 | 0.169924 | -0.052633 |
| `state_action` | `idm_gt_future` | 8 | `params/pose1/0/1` | `pose1_xyz` | 0.061483 | 0.169924 | -0.052633 |
| `state_action` | `idm_pred_future` | 0 | `params/pose0/0/0` | `pose0_xyz` | 0.053710 | 0.144647 | 0.039415 |
| `paper_state` | `idm_pred_future` | 0 | `params/pose0/0/0` | `pose0_xyz` | 0.048538 | 0.154485 | 0.038037 |
| `paper_state` | `idm_gt_future` | 1 | `params/pose0/0/1` | `pose0_xyz` | 0.046253 | 0.214663 | 0.036400 |
| `state_action` | `idm_gt_future` | 1 | `params/pose0/0/1` | `pose0_xyz` | 0.046253 | 0.214663 | 0.036400 |
| `paper_state` | `idm_gt_future` | 7 | `params/pose1/0/0` | `pose1_xyz` | 0.037481 | 0.147710 | -0.028718 |
| `state_action` | `idm_gt_future` | 7 | `params/pose1/0/0` | `pose1_xyz` | 0.037481 | 0.147710 | -0.028718 |
| `paper_state` | `idm_gt_future` | 0 | `params/pose0/0/0` | `pose0_xyz` | 0.028149 | 0.107313 | 0.020632 |
| `state_action` | `idm_gt_future` | 0 | `params/pose0/0/0` | `pose0_xyz` | 0.028149 | 0.107313 | 0.020632 |
| `state_action` | `idm_pred_future` | 2 | `params/pose0/0/2` | `pose0_xyz` | 0.005299 | 0.015028 | -0.001542 |
| `paper_state` | `idm_pred_future` | 2 | `params/pose0/0/2` | `pose0_xyz` | 0.005147 | 0.015534 | -0.001645 |
| `paper_state` | `idm_pred_future` | 9 | `params/pose1/0/2` | `pose1_xyz` | 0.004919 | 0.010985 | 0.004149 |
| `state_action` | `idm_pred_future` | 9 | `params/pose1/0/2` | `pose1_xyz` | 0.004529 | 0.010429 | 0.003222 |

## Interpretation

- This report identifies which encoded pick/place dimensions dominate IDM error.
- It does not prove execution repair by itself.
- Use the Phase3.8 repair probe for matched-prefix execution evidence.
