# Phase3.12b Condition Proxy / Score Ablation Preflight

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
| `import_ravens_tasks` | `True` |
| `import_ravens_environment` | `True` |
| `import_ccda_phase3_rollout` | `True` |
| `import_ccda_phase3_train_utils` | `True` |
| `import_phase3_policy_rollout` | `True` |
| `forbidden_modules_not_loaded` | `True` |
| `task_registered` | `True` |
| `task_instantiates` | `True` |
| `task_has_required_conditions` | `True` |
| `task_has_primary` | `True` |
| `windows_loadable` | `True` |
| `windows_conditions_match` | `True` |
| `windows_primary_match` | `True` |
| `windows_diagnostic_match` | `True` |
| `has_state_action_x` | `True` |
| `has_y_state` | `True` |
| `has_y_action` | `True` |
| `has_condition_name` | `True` |
| `has_split_name` | `True` |
| `has_th` | `True` |
| `has_action_dim` | `True` |
| `has_n_beads` | `True` |
| `y_action_dim_14` | `True` |
| `action_codec_dim_14` | `True` |
| `action_codec_no_camera_config` | `True` |
| `old_state_action_checkpoint_complete` | `True` |
| `phase39b_best_idm_exists` | `True` |
| `phase312_summary_exists` | `True` |
| `phase312_condition_upper_bound_supported` | `True` |
| `phase312_input_nearest_not_supported` | `True` |
| `phase312_compat_global_not_supported` | `True` |
| `phase312_no_phase4_scope` | `True` |
| `script_import_and_input_hazard_free` | `True` |
| `running_in_coord_bimanual` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `phase312_root_cause_label_misleading` | Phase3.12 table supports condition_nearest upper bound, not input-only simple locality. |

## Scope

- This preflight does not run rollout.
- Phase3.12b runs observable proxy and score-ablation diagnostics only.
- No model training.
- No future DDPM training.
- No Phase4 or CPS is allowed.
