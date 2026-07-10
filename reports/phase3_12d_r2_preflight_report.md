# Phase3.12d-r2 Paired-Visible Arming Preflight

## Verdict

- Verdict: `WARN`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Query-local snapshot allowed after environment audit: `True`

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
| `import_phase3_12c_matched_reset_common` | `True` |
| `import_phase3_12d_r1_common` | `True` |
| `import_phase3_12d_r2_common` | `True` |
| `forbidden_modules_not_loaded` | `True` |
| `task_deferred_arming_marker` | `True` |
| `task_arm_method` | `True` |
| `task_pending_state` | `True` |
| `task_armed_state` | `True` |
| `task_defer_env` | `True` |
| `task_post_arm_settle_env` | `True` |
| `task_breakaway_guard` | `True` |
| `task_zero_default_breakaway_damping` | `True` |
| `environment_step_lock_present` | `True` |
| `environment_physics_hook_present` | `True` |
| `environment_hook_error_surface_present` | `True` |
| `runtime_breakaway_damping_zero` | `True` |
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
| `legacy_checkpoint_bridge_required` | `True` |
| `r1_environment_audit_exists` | `True` |
| `r1_failed_for_visible_geometry` | `True` |
| `exists_phase3_12d_r1_common.py` | `True` |
| `exists_phase3_12d_r1_query_local_snapshot.py` | `True` |
| `exists_phase3_12d_r1_analyze.py` | `True` |
| `exists_phase3_12d_r2_common.py` | `True` |
| `exists_phase3_12d_r2_static_selftest.py` | `True` |
| `exists_phase3_12d_r2_environment_audit.py` | `True` |
| `exists_phase3_12d_r2_query_local_snapshot.py` | `True` |
| `exists_phase3_12d_r2_analyze.py` | `True` |
| `exists_phase3_12d_r2_postprocess.py` | `True` |
| `query_uses_r2_common` | `True` |
| `query_uses_deferred_arm_reset` | `True` |
| `query_does_not_use_r1_reset` | `True` |
| `query_one_worker_per_query_preserved` | `True` |
| `running_in_coord_bimanual` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `checkpoint_environment_semantics_changed` | Existing Phase3 windows/checkpoints were generated before deferred zero-offset arming. Negative candidate-headroom results require a repaired-data repeat before architecture-level conclusions. |

## Decision

- Any FAIL blocks the environment audit.
- The environment audit remains a separate hard gate before candidate execution.
- No training, Phase4, or CPS is permitted.
