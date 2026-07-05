# Phase3.4 Rollout Smoke Preflight

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
| `forbidden_modules_not_loaded_after_imports` | `True` |
| `task_registered` | `True` |
| `task_instantiates` | `True` |
| `task_conditions_exact_or_superset` | `True` |
| `task_has_hidden_breakaway_pin` | `True` |
| `windows_loadable` | `True` |
| `windows_conditions_match` | `True` |
| `windows_primary_match` | `True` |
| `windows_diagnostic_match` | `True` |
| `y_action_dim_14` | `True` |
| `action_template_exists` | `True` |
| `action_template_dim_14` | `True` |
| `action_template_no_camera_config` | `True` |
| `paper_state_checkpoints_exist` | `True` |
| `state_action_checkpoints_exist` | `True` |
| `phase33b_pass` | `True` |
| `rollout_script_hazard_free` | `True` |
| `running_in_coord_bimanual` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `PASS` | `none` | No preflight issues found. |

## Scope

- This preflight does not run rollout.
- TensorFlow / Ravens agents / Ravens models / Ravens datasets must remain unloaded.
- Rollout smoke requires explicit gates in the shell wrapper.
