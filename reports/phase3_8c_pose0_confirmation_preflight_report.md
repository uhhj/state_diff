# Phase3.8c Pose0 Confirmation Preflight

## Verdict

- Verdict: `PASS`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`

## Checks

| Check | Result |
|---|---:|
| `import_numpy` | `True` |
| `import_torch` | `True` |
| `import_pybullet` | `True` |
| `import_ravens_tasks` | `True` |
| `import_ravens_environment` | `True` |
| `import_phase3_8b_narrow_pose1_repair_probe` | `True` |
| `import_phase3_8b_analyze_narrow_pose1_repair` | `True` |
| `import_phase3_8_idm_geometry_repair_probe` | `True` |
| `forbidden_modules_not_loaded` | `True` |
| `task_registered` | `True` |
| `task_instantiates` | `True` |
| `task_has_required_conditions` | `True` |
| `task_has_hidden_breakaway_pin` | `True` |
| `windows_loadable` | `True` |
| `windows_conditions_match` | `True` |
| `windows_primary_match` | `True` |
| `windows_diagnostic_match` | `True` |
| `has_paper_x` | `True` |
| `has_state_action_x` | `True` |
| `has_y_action` | `True` |
| `y_action_dim_14` | `True` |
| `has_condition_name` | `True` |
| `has_visible_seed` | `True` |
| `has_window_t` | `True` |
| `has_source_file` | `True` |
| `has_all_confirmation_conditions` | `True` |
| `phase36_summary_exists` | `True` |
| `phase36_raw_prefix_available` | `True` |
| `phase36_state_match` | `True` |
| `phase37_summary_exists` | `True` |
| `phase37_root_geometry_blocker` | `True` |
| `phase38_sensitivity_exists` | `True` |
| `phase38_sensitivity_pass` | `True` |
| `phase38b_summary_exists` | `True` |
| `phase38b_pose0_supported` | `True` |
| `phase38b_warn_expected` | `True` |
| `phase38b_early_stop_expected` | `True` |
| `exists_phase3_8b_narrow_pose1_repair_probe.py` | `True` |
| `exists_phase3_8c_analyze_pose0_confirmation.py` | `True` |
| `exists_phase3_8c_run_pose0_confirmation.sh` | `True` |
| `script_import_and_input_hazard_free` | `True` |
| `running_in_coord_bimanual` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `PASS` | `none` | No preflight issues found. |

## Scope

- This preflight does not run confirmation execution.
- Phase3.8c requires explicit gates.
- No Phase4 or CPS is allowed.
