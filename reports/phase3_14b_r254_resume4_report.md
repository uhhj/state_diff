# Phase3.14b-r2.5.4 Resume4 Reproduction-Mismatch Audit

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r254_resume4_residual_reproduction_identity_unobservable`
- Static tests: `425 passed`
- Static-test scope: `27 frozen Phase3.14b r2.x files`
- Full-repository pytest required: `false`
- Retraining: `false`
- Robot-proxy attribution interpretable: `false`

## Conclusion

The report adapter is structurally consistent and every current-run training metric reproduces exactly, but the cross-device residual metrics do not reproduce and no residual initialization, final-state, prediction, optimizer, or loss-history fingerprints were persisted. The failure therefore cannot be classified as tiny numeric noise or as substantive scientific divergence from the committed evidence alone.

## Aggregate

| Item | Value |
|---|---:|
| Models failing historical reproduction | 3 |
| Changed Boolean contracts | 3 |
| Failed continuous metric checks | 15 |
| Adapter schema valid | true |
| Internal metric reproduction | true |
| Residual identity complete | false |

## Per-model reproduction checks

### `v_only_frozen_control`

| Check | Expected | Observed | Absolute diff | Relative diff | Tolerance | Pass |
|---|---:|---:|---:|---:|---:|---:|
| `contract.all_earlier_horizon_cable_geometry_pass` | `False` | `True` | n/a | n/a | exact | false |
| `contract.branch_transport_pass` | `True` | `True` | n/a | n/a | exact | true |
| `contract.final_horizon_cable_geometry_pass` | `True` | `True` | n/a | n/a | exact | true |
| `contract.historical_exact_reconstruction_pass` | `False` | `True` | n/a | n/a | exact | false |
| `contract.one_step_physical_pass` | `False` | `True` | n/a | n/a | exact | false |
| `contract.ordered_topology_pass` | `True` | `True` | n/a | n/a | exact | true |
| `state_group_z_metrics.cable_error_fraction` | `0.08426208800854225` | `0.08854592453283028` | 0.00428383652 | 0.05083943 | 2e-06 | false |
| `state_group_z_metrics.cable_z_mse` | `0.0006151102778066758` | `0.00013262675478221416` | 0.000482483523 | 0.784385403 | 2e-06 | false |
| `state_group_z_metrics.full_z_mse` | `0.00583997185300528` | `0.001198264114193443` | 0.00464170774 | 0.7948168 | 2e-06 | false |
| `state_group_z_metrics.robot_proxy_error_fraction` | `0.9157379119914577` | `0.9114540754671697` | 0.00428383652 | 0.00467801591 | 2e-06 | false |
| `state_group_z_metrics.robot_proxy_z_mse` | `0.026739418153799703` | `0.005460813551838358` | 0.0212786046 | 0.79577665 | 2e-06 | false |

### `ordered_mean_raw_g100`

| Check | Expected | Observed | Absolute diff | Relative diff | Tolerance | Pass |
|---|---:|---:|---:|---:|---:|---:|
| `contract.all_earlier_horizon_cable_geometry_pass` | `True` | `True` | n/a | n/a | exact | true |
| `contract.branch_transport_pass` | `True` | `True` | n/a | n/a | exact | true |
| `contract.final_horizon_cable_geometry_pass` | `True` | `True` | n/a | n/a | exact | true |
| `contract.historical_exact_reconstruction_pass` | `False` | `False` | n/a | n/a | exact | true |
| `contract.one_step_physical_pass` | `True` | `True` | n/a | n/a | exact | true |
| `contract.ordered_topology_pass` | `True` | `True` | n/a | n/a | exact | true |
| `state_group_z_metrics.cable_error_fraction` | `0.030343302156503674` | `0.0686348686383879` | 0.0382915665 | 1.26194461 | 2e-06 | false |
| `state_group_z_metrics.cable_z_mse` | `0.0001944287391684804` | `0.0004968533198503258` | 0.000302424581 | 1.55545205 | 2e-06 | false |
| `state_group_z_metrics.full_z_mse` | `0.005126106266632742` | `0.005791264174693068` | 0.000665157908 | 0.1297589 | 2e-06 | false |
| `state_group_z_metrics.robot_proxy_error_fraction` | `0.9696566978434963` | `0.9313651313616121` | 0.0382915665 | 0.039489818 | 2e-06 | false |
| `state_group_z_metrics.robot_proxy_z_mse` | `0.024852816376489794` | `0.02696890759406404` | 0.00211609122 | 0.0851449263 | 2e-06 | false |

### `ordered_cvar_contract_g010`

| Check | Expected | Observed | Absolute diff | Relative diff | Tolerance | Pass |
|---|---:|---:|---:|---:|---:|---:|
| `contract.all_earlier_horizon_cable_geometry_pass` | `True` | `True` | n/a | n/a | exact | true |
| `contract.branch_transport_pass` | `True` | `True` | n/a | n/a | exact | true |
| `contract.final_horizon_cable_geometry_pass` | `True` | `True` | n/a | n/a | exact | true |
| `contract.historical_exact_reconstruction_pass` | `False` | `False` | n/a | n/a | exact | true |
| `contract.one_step_physical_pass` | `True` | `True` | n/a | n/a | exact | true |
| `contract.ordered_topology_pass` | `True` | `True` | n/a | n/a | exact | true |
| `state_group_z_metrics.cable_error_fraction` | `0.07222416080139916` | `0.07062482275461884` | 0.00159933805 | 0.0221440863 | 2e-06 | false |
| `state_group_z_metrics.cable_z_mse` | `0.00025614462099049463` | `0.0006733762995476014` | 0.000417231679 | 1.62889104 | 2e-06 | false |
| `state_group_z_metrics.full_z_mse` | `0.002837218107052425` | `0.007627644482871997` | 0.00479042638 | 1.68842373 | 2e-06 | false |
| `state_group_z_metrics.robot_proxy_error_fraction` | `0.9277758391986007` | `0.9293751772453812` | 0.00159933805 | 0.00172384102 | 2e-06 | false |
| `state_group_z_metrics.robot_proxy_z_mse` | `0.013161512051300145` | `0.03544471721616958` | 0.0222832052 | 1.69305814 | 2e-06 | false |

## Confirmed source-code findings

- `robot_proxy_joint_selection_includes_fixed_joints`: task logs every URDF joint while control uses revolute env.joints. The pinned submodule remains unchanged in Resume4.
- `robot_proxy_end_effector_link_not_environment_tip_link`: task uses count-1 while environment owns ee_tip_link. The pinned submodule remains unchanged in Resume4.
- `robot_proxy_collection_errors_silently_zero_filled`: broad exception falls through to missing_zero_proxy. The pinned submodule remains unchanged in Resume4.

## Static-test scope boundary

The gate uses the exact frozen union of `tests/test_phase3_14b_r2*.py` and `tests/test_phase314b_r2*.py`. It does not use `--ignore`, `-k`, deselection, or a full-repository collection. The seven known out-of-scope collection failures are recorded as repository baseline defects and are not repaired in Resume4.

## Decision

Next: `Phase3.14b-r2.5.4 Resume5 same-device residual determinism and functional-reproduction contract audit`.

Until that audit completes, `train_only_recommendation`, `selected_configuration`, robot-metric interpretation, formal training, IDM, candidate execution, and Phase4/CPS remain blocked.
