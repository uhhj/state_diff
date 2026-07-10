# Phase3.12d-r2.1 Paired-Horizon Audit Preflight

## Verdict

- Verdict: `WARN`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`

## Checks

| Check | Result |
|---|---:|
| `import_numpy` | `True` |
| `import_torch` | `True` |
| `import_pybullet` | `True` |
| `import_scipy` | `True` |
| `import_ravens_tasks` | `True` |
| `import_ravens_environment` | `True` |
| `import_phase3_12d_r2_common` | `True` |
| `import_phase3_12d_r2_environment_audit` | `True` |
| `import_phase3_12d_r2_query_local_snapshot` | `True` |
| `import_phase3_12d_r2_analyze` | `True` |
| `import_phase3_12d_r21_paired_horizon_audit` | `True` |
| `forbidden_modules_not_loaded` | `True` |
| `exists_phase3_12d_r2_common_py` | `True` |
| `exists_phase3_12d_r2_environment_audit_py` | `True` |
| `exists_phase3_12d_r2_query_local_snapshot_py` | `True` |
| `exists_phase3_12d_r2_analyze_py` | `True` |
| `exists_phase3_12d_r2_postprocess_py` | `True` |
| `exists_phase3_12d_r21_paired_horizon_audit_py` | `True` |
| `exists_phase3_12d_r21_static_selftest_py` | `True` |
| `exists_phase3_12d_r21_run_sh` | `True` |
| `r2_historical_absolute_drift_comparator_present` | `True` |
| `r2_common_exports_post_arm_xy` | `True` |
| `r2_common_exports_post_arm_velocity` | `True` |
| `r2_common_exports_pre_arm_xy` | `True` |
| `r2_common_uses_direct_physics_steps` | `True` |
| `r2_runner_hardcodes_old_environment_audit` | `True` |
| `r2_summary_exists` | `True` |
| `r2_summary_failed` | `True` |
| `r2_query_matrix_was_blocked` | `True` |
| `submodule_clean` | `True` |
| `running_in_coord_bimanual` | `True` |
| `r21_runner_does_not_patch_submodule` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `r2_post_arm_gate_compares_hidden_horizon_to_free_t0` | Historical r2 report cannot establish condition-specific excess drift. |
| `WARN` | `do_not_run_original_r2_runner_for_r21` | The original runner re-executes the historical comparator and will block again. |

## Decision

- The historical r2 absolute-drift failure is not condition-specific evidence.
- r2.1 compares free and hidden at the same no-action horizon.
- r2.1 does not modify the DeformableRavens submodule.
- No training, Phase4, or CPS is allowed.
