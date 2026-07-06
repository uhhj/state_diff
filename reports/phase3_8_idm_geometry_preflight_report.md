# Phase3.8 IDM Geometry Preflight

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
| `has_y_state_or_y_final` | `True` |
| `has_condition_name` | `True` |
| `has_visible_seed` | `True` |
| `has_window_t` | `True` |
| `has_source_file` | `True` |
| `y_action_dim_14` | `True` |
| `action_template_exists` | `True` |
| `action_codec_dim_14` | `True` |
| `action_codec_no_camera_config` | `True` |
| `paper_state_checkpoints_complete` | `True` |
| `state_action_checkpoints_complete` | `True` |
| `phase37_summary_exists` | `True` |
| `phase37_expected_root_cause` | `True` |
| `phase37_verdict_fail_expected` | `True` |
| `phase37_gt_progress_positive` | `True` |
| `phase37_idm_gt_no_progress` | `True` |
| `phase36_summary_exists` | `True` |
| `phase36_raw_prefix_available` | `True` |
| `phase36_state_match` | `True` |
| `script_import_and_input_hazard_free` | `True` |
| `running_in_coord_bimanual` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `PASS` | `none` | No preflight issues found. |

## Scope

- This preflight does not run repair probes.
- Phase3.8 requires explicit gates.
- No Phase4 or CPS is allowed.
