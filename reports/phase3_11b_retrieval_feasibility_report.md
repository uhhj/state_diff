# Phase3.11b Future Target Compatibility + Retrieved Action Feasibility Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase311b_live_source_context_mismatch_supported`
- Rows: `63`
- Progress status: `completed`

## Per-Condition Compatibility Diagnosis

| Condition | Source GT Δ | Source repaired Δ | Live GT Δ | Live repaired Δ | Live DDPM Δ | Live condition retrieval Δ | Source GT effective | Source repaired effective | Live GT effective | Live repaired effective | Live context L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free` | 0.0833 | 0.1528 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `True` | `True` | `False` | `False` | 0.1635 |
| `hidden_breakaway_pin` | 0.1250 | 0.1111 | 0.0000 | 0.0000 | 0.0000 | 0.1111 | `True` | `True` | `False` | `False` | 0.1342 |
| `hidden_high_friction` | 0.1528 | 0.1250 | 0.0000 | -0.1667 | 0.0278 | 0.0000 | `True` | `True` | `False` | `False` | 0.0782 |

## Variant Summary

| Condition | Variant | Rows | OK | Timeout | Delta | Final after | Pull | Action OOD | Context L2 | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free` | `source_gt_action` | 3 | 3 | 0 | 0.0833 | 0.0833 | 0.0939 | nan | nan | 0 |
| `free` | `source_old_idm_future` | 3 | 3 | 0 | 0.0833 | 0.0833 | 0.1069 | 0.453 | nan | 0 |
| `free` | `source_repaired_idm_future` | 3 | 3 | 0 | 0.1528 | 0.1528 | 0.0948 | 0.584 | nan | 0 |
| `free` | `live_retrieved_gt_action` | 3 | 3 | 0 | 0.0000 | 0.1111 | 0.0939 | nan | 0.1261 | 0 |
| `free` | `live_repaired_idm_retrieved_future` | 3 | 3 | 0 | 0.0000 | 0.0000 | 0.2572 | 0.642 | 0.1635 | 0 |
| `free` | `live_ddpm_mean_repaired_idm` | 3 | 3 | 0 | 0.0000 | 0.0000 | 0.2520 | 0.603 | 0.1434 | 0 |
| `free` | `live_condition_retrieval_repaired_idm` | 3 | 3 | 0 | 0.0000 | 0.0694 | 0.2613 | 0.470 | 0.1113 | 0 |
| `hidden_breakaway_pin` | `source_gt_action` | 3 | 3 | 0 | 0.1250 | 0.1389 | 0.1501 | nan | nan | 0 |
| `hidden_breakaway_pin` | `source_old_idm_future` | 3 | 3 | 0 | 0.0833 | 0.0972 | 0.1523 | 0.480 | nan | 0 |
| `hidden_breakaway_pin` | `source_repaired_idm_future` | 3 | 3 | 0 | 0.1111 | 0.1250 | 0.1473 | 0.517 | nan | 0 |
| `hidden_breakaway_pin` | `live_retrieved_gt_action` | 3 | 3 | 0 | 0.0000 | 0.0694 | 0.1501 | nan | 0.1140 | 0 |
| `hidden_breakaway_pin` | `live_repaired_idm_retrieved_future` | 3 | 3 | 0 | 0.0000 | 0.0000 | 0.1983 | 0.692 | 0.1342 | 0 |
| `hidden_breakaway_pin` | `live_ddpm_mean_repaired_idm` | 3 | 3 | 0 | 0.0000 | 0.1250 | 0.2546 | 0.587 | 0.1333 | 0 |
| `hidden_breakaway_pin` | `live_condition_retrieval_repaired_idm` | 3 | 3 | 0 | 0.1111 | 0.2778 | 0.3247 | 0.602 | 0.1319 | 0 |
| `hidden_high_friction` | `source_gt_action` | 3 | 3 | 0 | 0.1528 | 0.1528 | 0.0877 | nan | nan | 0 |
| `hidden_high_friction` | `source_old_idm_future` | 3 | 3 | 0 | 0.0000 | 0.0000 | 0.0978 | 0.324 | nan | 0 |
| `hidden_high_friction` | `source_repaired_idm_future` | 3 | 3 | 0 | 0.1250 | 0.1250 | 0.0897 | 0.352 | nan | 0 |
| `hidden_high_friction` | `live_retrieved_gt_action` | 3 | 3 | 0 | 0.0000 | 0.0417 | 0.0877 | nan | 0.1251 | 0 |
| `hidden_high_friction` | `live_repaired_idm_retrieved_future` | 3 | 3 | 0 | -0.1667 | 0.0417 | 0.1452 | 0.483 | 0.0782 | 0 |
| `hidden_high_friction` | `live_ddpm_mean_repaired_idm` | 3 | 3 | 0 | 0.0278 | 0.1944 | 0.1191 | 0.355 | 0.1174 | 0 |
| `hidden_high_friction` | `live_condition_retrieval_repaired_idm` | 3 | 3 | 0 | 0.0000 | 0.0139 | 0.2811 | 0.570 | 0.1471 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `retrieved_action_transfer_to_live_context_fails` | {'condition': 'hidden_breakaway_pin', 'source_gt_delta': 0.125, 'source_old_idm_delta': 0.08333333333333333, 'source_repaired_idm_delta': 0.1111111111111111, 'live_retrieved_gt_delta': 0.0, 'live_repaired_retrieved_delta': 0.0, 'live_ddpm_mean_delta': 0.0, 'live_condition_retrieval_delta': 0.11111111111111112, 'source_gt_pull': 0.15012511486808458, 'source_repaired_pull': 0.14730341359972954, 'live_retrieved_gt_pull': 0.15012511486808458, 'live_repaired_retrieved_pull': 0.1982981413602829, 'live_ddpm_mean_pull': 0.2546310101946195, 'live_context_l2': 0.1342360700170199, 'source_gt_effective': True, 'source_repaired_effective': True, 'live_gt_effective': False, 'live_repaired_effective': False} |
| `WARN` | `repaired_idm_retrieved_future_transfer_to_live_context_fails` | {'condition': 'hidden_breakaway_pin', 'source_gt_delta': 0.125, 'source_old_idm_delta': 0.08333333333333333, 'source_repaired_idm_delta': 0.1111111111111111, 'live_retrieved_gt_delta': 0.0, 'live_repaired_retrieved_delta': 0.0, 'live_ddpm_mean_delta': 0.0, 'live_condition_retrieval_delta': 0.11111111111111112, 'source_gt_pull': 0.15012511486808458, 'source_repaired_pull': 0.14730341359972954, 'live_retrieved_gt_pull': 0.15012511486808458, 'live_repaired_retrieved_pull': 0.1982981413602829, 'live_ddpm_mean_pull': 0.2546310101946195, 'live_context_l2': 0.1342360700170199, 'source_gt_effective': True, 'source_repaired_effective': True, 'live_gt_effective': False, 'live_repaired_effective': False} |

## Interpretation

- This audit tests whether retrieved future/action is executable in its source prefix and transferable to current live prefix.
- Retrieved y_state / y_action / condition labels are diagnostic only and not deployable policy inputs.
- No model training was run.
- No future DDPM was trained.
- No Phase4 or CPS was run.
