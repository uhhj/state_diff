# Phase3.6 Window Schema Audit

## Verdict

- Verdict: `PASS`
- condition_key: `condition_name`
- visible_seed_key: `visible_seed`
- window_t_key: `window_t`
- source_key: `source_file`

## Checks

| Check | Result |
|---|---:|
| `windows_exists` | `True` |
| `has_paper_x` | `True` |
| `has_state_action_x` | `True` |
| `has_y_action` | `True` |
| `has_condition_key` | `True` |
| `has_visible_seed_key` | `True` |
| `has_window_t_key` | `True` |
| `has_source_key` | `True` |
| `meta_conditions_match` | `True` |
| `meta_primary_match` | `True` |
| `meta_diagnostic_match` | `True` |
| `y_action_dim_14` | `True` |
| `state_action_extra_dim` | `42` |
| `state_action_extra_dim_multiple_of_14` | `True` |
| `can_use_state_action_tail_as_low_conf_prefix` | `True` |

## Key Shapes

| Key | Shape | Preview |
|---|---|---|
| `paper_x` | `[956, 405]` | `['0.6833013', '0.45117375', '0.69494087', '0.45916364', '0.6943897']` |
| `state_action_x` | `[956, 447]` | `['0.6833013', '0.45117375', '0.69494087', '0.45916364', '0.6943897']` |
| `y_state` | `[956, 4, 135]` | `['0.6020681', '0.23566061', '0.5942435', '0.22384235', '0.58630484']` |
| `y_final_state` | `[956, 135]` | `['0.6188594', '0.18970697', '0.60672563', '0.18237871', '0.5929038']` |
| `y_action` | `[956, 14]` | `['0.66875', '0.40625', '0.03550928', '0.0', '0.0']` |
| `condition_id` | `[956]` | `['0', '1', '2', '3', '0']` |
| `condition_name` | `[956]` | `['free', 'hidden_pin', 'hidden_high_friction', 'hidden_breakaway_pin', 'free']` |
| `visible_seed` | `[956]` | `['1000', '1000', '1000', '1000', '1000']` |
| `split_name` | `[956]` | `['train', 'train', 'train', 'train', 'train']` |
| `source_file` | `[956]` | `['000000-9.pkl', '000000-3.pkl', '000000-5.pkl', '000000-4.pkl', '000000-9.pkl']` |
| `window_t` | `[956]` | `['0', '0', '0', '0', '1']` |
| `episode_action_len` | `[956]` | `['9', '3', '5', '4', '9']` |
| `success` | `[956]` | `['1', '0', '1', '1', '1']` |
| `final_fraction` | `[956]` | `['1.0', '0.0', '1.0', '1.0', '1.0']` |
| `state_dim` | `[]` | `['135']` |
| `robot_pose_dim` | `[]` | `['39']` |
| `action_dim` | `[]` | `['14']` |
| `n_beads` | `[]` | `['24']` |
| `th` | `[]` | `['3']` |
| `tf` | `[]` | `['4']` |
| `action_template_json_or_pickle_path` | `[]` | `['/data/state_diff2/data/phase3_state_diff_windows/phase3_action_template.pkl']` |
| `robot_pose_proxy_source` | `[956]` | `['pybullet_robot_body', 'pybullet_robot_body', 'pybullet_robot_body', 'pybullet_robot_body', 'pybullet_robot_body']` |
| `meta_json` | `[]` | `['{"action_codec": {"class": "ExecutableActionCodec", "dim": 14, "num_camera_config_paths": 0, "num_param_paths": 14, "num_paths": 14, "paths": ["params/pose0/0/0", "params/pose0/0/1", "params/pose0/0/2", "params/pose0/1/0", "params/pose0/1/1", "params/pose0/1/2", "params/pose0/1/3", "params/pose1/0/0", "params/pose1/0/1", "params/pose1/0/2", "params/pose1/1/0", "params/pose1/1/1", "params/pose1/1/2", "params/pose1/1/3"]}, "action_codec_summary": {"class": "ExecutableActionCodec", "dim": 14, "num_camera_config_paths": 0, "num_param_paths": 14, "num_paths": 14, "paths": ["params/pose0/0/0", "params/pose0/0/1", "params/pose0/0/2", "params/pose0/1/0", "params/pose0/1/1", "params/pose0/1/2", "params/pose0/1/3", "params/pose1/0/0", "params/pose1/0/1", "params/pose1/0/2", "params/pose1/1/0", "params/pose1/1/1", "params/pose1/1/2", "params/pose1/1/3"]}, "action_dim": 14, "action_template_json_or_pickle_path": "/data/state_diff2/data/phase3_state_diff_windows/phase3_action_template.pkl", "action_template_path": "/data/state_diff2/data/phase3_state_diff_windows/phase3_action_template.pkl", "canonicalization_rule": "group by split_name, visible_seed, window_t; copy free input to paired hidden branches; targets remain condition-specific", "canonicalization_summary_path": "/data/state_diff2/reports/phase3_canonicalization_summary.json", "canonicalized_paired_input_windows": 717, "ccda_breakaway_bead_ratio": "0.45", "ccda_breakaway_disp": "0.045", "ccda_breakaway_force": "2.6", "ccda_oracle_breakaway_pull_dist": "0.36", "complete_condition_filter": "keep only split/visible_seed/window_t groups containing: free, hidden_pin, hidden_high_friction, hidden_breakaway_pin", "conditions": ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"], "diagnostic_branch_pair": "free_vs_hidden_pin", "diagnostic_hidden_condition": "hidden_pin", "dropped_windows_incomplete_condition_groups": 1921, "episode_action_len_max": 20, "episode_action_len_mean": 7.076359832635983, "episode_action_len_min": 1, "feature_schema": {"forbidden_not_in_x": ["hidden_condition", "hidden_contact_meta", "recoverability_params", "recoverability_class_candidate", "breakaway_released", "breakaway_release_step", "breakaway_step", "breakaway_threshold", "breakaway_max_disp_seen", "condition_id", "condition_name", "condition_label", "condition_id_onehot", "success", "final_fraction", "ccda_pair_group", "pair_group", "source_file", "visible_seed", "ccda_visible_seed"], "paper_x": ["bead_xy_history", "bead_velocity_history", "robot_pose_proxy_history"], "state_action_x": ["paper_x", "past_action_history_excluding_current_action"], "y_action": ["current_executable_pick_place_action"], "y_state": ["future_state_trajectory"]}, "forbidden_metadata_not_in_x": ["hidden_condition", "hidden_contact_meta", "recoverability_params", "recoverability_class_candidate", "breakaway_released", "breakaway_release_step", "breakaway_step", "breakaway_threshold", "breakaway_max_disp_seen", "condition_id", "condition_name", "condition_label", "condition_id_onehot", "success", "final_fraction", "ccda_pair_group", "pair_group", "source_file", "visible_seed", "ccda_visible_seed"], "heldout_visible_seed_count": 30, "heldout_visible_seeds": [100000, 100001, 100002, 100003, 100004, 100005, 100006, 100007, 100008, 100009, 100010, 100011, 100012, 100013, 100014, 100015, 100016, 100017, 100018, 100019, 100020, 100021, 100022, 100023, 100024, 100025, 100026, 100027, 100028, 100029], "max_windows_per_episode": 0, "n_beads": 24, "num_episodes_loaded": 520, "num_heldout_windows": 188, "num_train_windows": 768, "num_windows": 956, "paper_x_shape": [956, 405], "phase2_5_selected_config": "breakaway_force_2p6_disp_0p045_pull_0p36", "phase2_5_selected_recoverable_condition": "hidden_breakaway_pin", "primary_branch_pair": "free_vs_hidden_breakaway_pin", "primary_hidden_condition": "hidden_breakaway_pin", "raw_num_windows_before_complete_condition_filter": 2877, "robot_pose_dim": 39, "robot_pose_proxy_source_counts": {"pybullet_robot_body": 13471}, "state_action_extra_all_zero": false, "state_action_extra_std": 0.2338666021823883, "state_action_x_shape": [956, 447], "state_dim": 135, "tf": 4, "th": 3, "train_heldout_seed_overlap": [], "train_visible_seed_count": 100, "train_visible_seeds": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099], "windows_per_condition": {"free": 239, "hidden_breakaway_pin": 239, "hidden_high_friction": 239, "hidden_pin": 239}, "y_action_shape": [956, 14]}']` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `PASS` | `none` | No schema issues found. |

## Interpretation

- Raw source/action-file prefix is high-confidence matched replay.
- state_action_x tail prefix is low-confidence because canonicalization may have copied free inputs.
- If no prefix source exists, Phase3.6 must report BLOCKED instead of claiming codec failure.
