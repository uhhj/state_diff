# Phase3.5 Action Execution Preflight

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
| `y_action_dim_14` | `True` |
| `action_template_exists` | `True` |
| `action_template_dim_14` | `True` |
| `action_template_no_camera_config` | `True` |
| `paper_state_checkpoints_complete` | `True` |
| `state_action_checkpoints_complete` | `True` |
| `phase34_exists` | `True` |
| `phase34_warn_or_pass` | `True` |
| `phase34_rows_positive` | `True` |
| `policy_script_no_hazard` | `True` |
| `running_in_coord_bimanual` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `PASS` | `none` | No preflight issues found. |

## Scope

- This preflight does not execute actions.
- Phase3.5 action diagnostic requires explicit gates.
- No Phase4 or CPS is allowed.
