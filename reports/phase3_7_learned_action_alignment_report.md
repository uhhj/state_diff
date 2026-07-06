# Phase3.7 Learned Action Alignment Report

## Verdict

- Verdict: `FAIL`
- Root cause: `idm_action_execution_geometry_blocker`
- Rows: `96`
- IDM(GT future) mean action MAE to GT: `0.012991`
- IDM(pred future) mean action MAE to GT: `0.028539`
- Mean future error MAE: `0.069576`
- GT exec mean Δ final_fraction: `0.208333`
- Oracle exec mean Δ final_fraction: `0.125000`
- IDM(GT future) exec mean Δ final_fraction: `0.000000`
- IDM(pred future) exec mean Δ final_fraction: `0.005682`

## Summary Table

| Baseline | Source | Condition | Rows | Action MAE | Action L2 | Cosine | Pose0 dist | Pose1 dist | Pull angle | Future MAE | OOD | Clip | Exec success | Exec Δ final_fraction | Prefix MAE | Failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_state` | `gt_y_action` | `free` | 3 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00 | 0.0549 | nan | 0.0000 | 0.000 | 0.2917 | 0.0652 | 0 |
| `paper_state` | `gt_y_action` | `hidden_breakaway_pin` | 3 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00 | 0.0510 | nan | 0.0000 | 0.000 | 0.1667 | 0.0652 | 0 |
| `paper_state` | `gt_y_action` | `hidden_high_friction` | 3 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00 | 0.0532 | nan | 0.0000 | 0.667 | 0.9167 | 0.0331 | 0 |
| `paper_state` | `gt_y_action` | `hidden_pin` | 3 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00 | 0.1197 | nan | 0.0000 | 0.000 | 0.0000 | 0.0597 | 0 |
| `paper_state` | `idm_gt_future` | `free` | 3 | 0.0158 | 0.1327 | 0.9953 | 0.1194 | 0.0434 | 23.37 | 0.0549 | 0.7539 | 0.0000 | 0.000 | 0.0000 | 0.0652 | 0 |
| `paper_state` | `idm_gt_future` | `hidden_breakaway_pin` | 3 | 0.0132 | 0.1128 | 0.9971 | 0.0418 | 0.0985 | 18.97 | 0.0510 | 0.6963 | 0.0000 | 0.000 | 0.0000 | 0.0651 | 0 |
| `paper_state` | `idm_gt_future` | `hidden_high_friction` | 3 | 0.0114 | 0.0930 | 0.9980 | 0.0331 | 0.0822 | 21.00 | 0.0532 | 0.6545 | 0.0000 | 0.000 | 0.0000 | 0.0189 | 0 |
| `paper_state` | `idm_gt_future` | `hidden_pin` | 3 | 0.0117 | 0.0982 | 0.9974 | 0.0303 | 0.0893 | 3.93 | 0.1197 | 0.7728 | 0.0000 | 0.000 | 0.0000 | 0.0609 | 0 |
| `paper_state` | `idm_pred_future` | `free` | 3 | 0.0322 | 0.2789 | 0.9853 | 0.0944 | 0.2504 | 34.70 | 0.0549 | 0.7189 | 0.0000 | 0.000 | 0.0000 | 0.0651 | 0 |
| `paper_state` | `idm_pred_future` | `hidden_breakaway_pin` | 3 | 0.0270 | 0.2317 | 0.9905 | 0.0748 | 0.2177 | 37.40 | 0.0510 | 0.6799 | 0.0000 | 0.000 | 0.0000 | 0.0651 | 0 |
| `paper_state` | `idm_pred_future` | `hidden_high_friction` | 3 | 0.0209 | 0.1823 | 0.9930 | 0.0426 | 0.1737 | 27.21 | 0.0532 | 0.6675 | 0.0000 | 0.333 | -0.0208 | 0.0236 | 0 |
| `paper_state` | `idm_pred_future` | `hidden_pin` | 3 | 0.0408 | 0.3415 | 0.9784 | 0.2402 | 0.2241 | 15.10 | 0.1197 | 0.7727 | 0.0000 | 0.000 | 0.0000 | 0.0605 | 0 |
| `paper_state` | `same_state_oracle` | `free` | 3 | nan | nan | nan | 0.3280 | 0.1644 | 45.20 | 0.0549 | nan | nan | 0.000 | 0.2222 | 0.0651 | 0 |
| `paper_state` | `same_state_oracle` | `hidden_breakaway_pin` | 3 | nan | nan | nan | 0.2091 | 0.2855 | 54.85 | 0.0510 | nan | nan | 0.000 | 0.2361 | 0.0652 | 0 |
| `paper_state` | `same_state_oracle` | `hidden_high_friction` | 3 | nan | nan | nan | nan | nan | nan | 0.0532 | nan | nan | 0.667 | 0.0000 | 0.0336 | 0 |
| `paper_state` | `same_state_oracle` | `hidden_pin` | 3 | nan | nan | nan | nan | nan | nan | 0.1197 | nan | nan | 0.000 | 0.0000 | 0.0574 | 0 |
| `state_action` | `gt_y_action` | `free` | 3 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00 | 0.0543 | nan | 0.0000 | 0.000 | 0.2778 | 0.0651 | 0 |
| `state_action` | `gt_y_action` | `hidden_breakaway_pin` | 3 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00 | 0.0513 | nan | 0.0000 | 0.000 | 0.2222 | 0.0651 | 0 |
| `state_action` | `gt_y_action` | `hidden_high_friction` | 3 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00 | 0.0509 | nan | 0.0000 | 0.000 | 0.2639 | 0.0078 | 0 |
| `state_action` | `gt_y_action` | `hidden_pin` | 3 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.00 | 0.1214 | nan | 0.0000 | 0.000 | 0.0000 | 0.0576 | 0 |
| `state_action` | `idm_gt_future` | `free` | 3 | 0.0158 | 0.1327 | 0.9953 | 0.1194 | 0.0434 | 23.37 | 0.0543 | 0.7539 | 0.0000 | 0.000 | 0.0000 | 0.0651 | 0 |
| `state_action` | `idm_gt_future` | `hidden_breakaway_pin` | 3 | 0.0132 | 0.1128 | 0.9971 | 0.0418 | 0.0985 | 18.97 | 0.0513 | 0.6963 | 0.0000 | 0.000 | 0.0139 | 0.0652 | 0 |
| `state_action` | `idm_gt_future` | `hidden_high_friction` | 3 | 0.0114 | 0.0930 | 0.9980 | 0.0331 | 0.0822 | 21.00 | 0.0509 | 0.6545 | 0.0000 | 0.000 | -0.0139 | 0.0082 | 0 |
| `state_action` | `idm_gt_future` | `hidden_pin` | 3 | 0.0117 | 0.0982 | 0.9974 | 0.0303 | 0.0893 | 3.93 | 0.1214 | 0.7728 | 0.0000 | 0.000 | 0.0000 | 0.0590 | 0 |
| `state_action` | `idm_pred_future` | `free` | 3 | 0.0289 | 0.2314 | 0.9900 | 0.0875 | 0.2038 | 38.08 | 0.0543 | 0.6508 | 0.0000 | 0.000 | 0.0000 | 0.0652 | 0 |
| `state_action` | `idm_pred_future` | `hidden_breakaway_pin` | 3 | 0.0246 | 0.2332 | 0.9902 | 0.0656 | 0.2188 | 17.05 | 0.0513 | 0.6735 | 0.0000 | 0.000 | 0.0000 | 0.0652 | 0 |
| `state_action` | `idm_pred_future` | `hidden_high_friction` | 3 | 0.0186 | 0.1818 | 0.9934 | 0.0392 | 0.1751 | 16.09 | 0.0509 | 0.6812 | 0.0000 | 0.333 | 0.0833 | 0.0281 | 0 |
| `state_action` | `idm_pred_future` | `hidden_pin` | 3 | 0.0353 | 0.2983 | 0.9844 | 0.2065 | 0.1900 | 8.61 | 0.1214 | 0.6848 | 0.0000 | 0.000 | 0.0000 | 0.0576 | 0 |
| `state_action` | `same_state_oracle` | `free` | 3 | nan | nan | nan | 0.3280 | 0.1644 | 45.20 | 0.0543 | nan | nan | 0.000 | 0.2361 | 0.0652 | 0 |
| `state_action` | `same_state_oracle` | `hidden_breakaway_pin` | 3 | nan | nan | nan | 0.2091 | 0.2855 | 54.85 | 0.0513 | nan | nan | 0.000 | 0.2361 | 0.0652 | 0 |
| `state_action` | `same_state_oracle` | `hidden_high_friction` | 3 | nan | nan | nan | 0.1769 | 0.0303 | 92.76 | 0.0509 | nan | nan | 0.333 | -0.0139 | 0.0180 | 0 |
| `state_action` | `same_state_oracle` | `hidden_pin` | 3 | nan | nan | nan | nan | nan | nan | 0.1214 | nan | nan | 0.000 | 0.0000 | 0.0564 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `future_prediction_degrades_idm_action` | IDM(pred future) worse than IDM(GT future): gt_mae=0.012991, pred_mae=0.028539, gt_pose1=0.078348, pred_pose1=0.206699 |
| `FAIL` | `idm_gt_future_executes_poorly_despite_gt_progress` | GT action progresses but IDM(GT future) does not: gt_delta=0.208333, idm_gt_delta=0.000000 |

## Interpretation

- If IDM(GT future) is far from GT y_action, the first blocker is IDM teacher-forced alignment.
- If IDM(GT future) is close but IDM(pred future) is far, the first blocker is future DDPM to action alignment.
- If both are close offline but closed-loop rollout failed, the likely blocker is live closed-loop history update.
- This is not Phase4 and not CPS evidence.
