# Phase3.9b Geometry IDM Ablation Preflight

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
| `import_ccda_phase3_models` | `True` |
| `import_ccda_phase3_train_utils` | `True` |
| `import_phase3_9_train_geometry_idm` | `True` |
| `import_phase3_9_controlled_retry` | `True` |
| `import_phase3_9_analyze_controlled_retry` | `True` |
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
| `has_split_name` | `True` |
| `action_codec_dim_14` | `True` |
| `action_codec_no_camera_config` | `True` |
| `phase39_summary_exists` | `True` |
| `phase39_repair_supported` | `True` |
| `phase39_primary_improved` | `True` |
| `phase39_hidden_breakaway_improved` | `True` |
| `phase39_train_summary_exists` | `True` |
| `phase39_train_inverse_only` | `True` |
| `script_import_and_input_hazard_free` | `True` |
| `running_in_coord_bimanual` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `PASS` | `none` | No preflight issues found. |

## Scope

- This preflight does not train.
- Phase3.9b trains inverse dynamics ablations only.
- No future DDPM training.
- No Phase4 or CPS is allowed.
