# Phase3.6 Matched Replay Preflight

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
| `has_y_action` | `True` |
| `y_action_dim_14` | `True` |
| `action_template_exists` | `True` |
| `action_template_dim_14` | `True` |
| `action_template_no_camera_config` | `True` |
| `schema_audit_exists` | `True` |
| `schema_audit_not_fail` | `True` |
| `has_visible_seed_key` | `True` |
| `has_window_t_key` | `True` |
| `has_condition_key` | `True` |
| `has_high_conf_source_key` | `True` |
| `can_low_conf_prefix` | `True` |
| `has_any_prefix_source` | `True` |
| `phase35_exists` | `True` |
| `phase35_verdict_fail_expected` | `True` |
| `phase35_root_cause_recorded` | `True` |
| `script_import_hazard_free` | `True` |
| `running_in_coord_bimanual` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `PASS` | `none` | No preflight issues found. |

## Scope

- This preflight does not execute matched replay.
- Matched replay requires explicit gates.
- No Phase4 or CPS is allowed.
