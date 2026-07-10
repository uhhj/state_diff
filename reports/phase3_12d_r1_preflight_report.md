# Phase3.12d-r1 Query-Local Snapshot Preflight

## Verdict

- Verdict: `WARN`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Query-local snapshot allowed: `True`

## Checks

| Check | Result |
|---|---:|
| `import_numpy` | `True` |
| `import_torch` | `True` |
| `import_pybullet` | `True` |
| `import_scipy` | `True` |
| `import_ravens_tasks` | `True` |
| `import_ravens_environment` | `True` |
| `import_ccda_phase3_rollout` | `True` |
| `import_ccda_phase3_data_io` | `True` |
| `import_ccda_phase3_train_utils` | `True` |
| `import_phase3_policy_rollout` | `True` |
| `import_phase3_12b_proxy_score_rollout` | `True` |
| `import_phase3_12c_matched_reset_common` | `True` |
| `import_phase3_12d_r1_common` | `True` |
| `forbidden_modules_not_loaded` | `True` |
| `windows_loadable` | `True` |
| `has_state_action_x` | `True` |
| `has_y_state` | `True` |
| `has_y_action` | `True` |
| `has_condition_name` | `True` |
| `has_split_name` | `True` |
| `has_th` | `True` |
| `has_action_dim` | `True` |
| `has_n_beads` | `True` |
| `action_dim_14` | `True` |
| `y_action_dim_14` | `True` |
| `codec_dim_14` | `True` |
| `codec_no_camera_config` | `True` |
| `state_checkpoint_exists` | `True` |
| `repaired_idm_exists` | `True` |
| `phase312c_summary_exists` | `True` |
| `phase312c_matched_reset_passed` | `True` |
| `phase312d_failure_is_engineering_not_scientific` | `True` |
| `environment_step_lock_present` | `True` |
| `environment_physics_hook_present` | `True` |
| `environment_hook_error_surface_present` | `True` |
| `task_physics_hook_present` | `True` |
| `task_breakaway_physics_counter_present` | `True` |
| `task_breakaway_release_physics_step_present` | `True` |
| `task_breakaway_constraint_cleared_on_release` | `True` |
| `query_uses_one_worker_per_query` | `True` |
| `query_uses_save_state` | `True` |
| `query_uses_restore_state` | `True` |
| `query_does_not_cross_process_replay_candidates` | `True` |
| `action_history_appended_after_step` | `True` |
| `condition_label_only_in_diagnostic_candidates` | `True` |
| `running_in_coord_bimanual` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `prior_phase312d_invalid_due_to_prefix_replay` | {'verdict': 'FAIL', 'root_cause': 'phase312d_code_or_prefix_replay_integrity_failed'} |

## Decision

- Any FAIL blocks environment audit and candidate execution.
- The prior Phase3.12d report is invalid scientific evidence because prefix replay was incomplete.
- No model training, Phase4, or CPS is permitted.
