# Phase3.10b Future Rollout Audit Preflight

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
| `import_ccda_phase3_rollout` | `True` |
| `import_ccda_phase3_train_utils` | `True` |
| `import_phase3_policy_rollout` | `True` |
| `import_phase3_10_controlled_learned_rollout` | `True` |
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
| `y_action_dim_14` | `True` |
| `has_condition_name` | `True` |
| `has_split_name` | `True` |
| `has_th` | `True` |
| `has_action_dim` | `True` |
| `has_n_beads` | `True` |
| `action_codec_dim_14` | `True` |
| `action_codec_no_camera_config` | `True` |
| `old_state_action_checkpoint_complete` | `True` |
| `phase39_default_idm_exists` | `True` |
| `phase39b_best_idm_exists` | `True` |
| `phase39b_summary_exists` | `True` |
| `phase39b_best_matches` | `True` |
| `phase39b_default_robust` | `True` |
| `phase310_summary_exists` | `True` |
| `phase310_repaired_not_supported` | `True` |
| `phase310_no_primary_improvement` | `True` |
| `script_import_and_input_hazard_free` | `True` |
| `running_in_coord_bimanual` | `True` |

## Checkpoints

| Policy | Path |
|---|---|
| `old_state_action` | `/data/state_diff2/checkpoints/phase3/state_action/fold_0_seed_0` |
| `phase39_default_geometry` | `/data/state_diff2/checkpoints/phase3_9_geometry_idm/state_action/fold_phase3_9_seed_390000/inverse_dynamics.pt` |
| `phase39b_xy_only_high_weight` | `/data/state_diff2/checkpoints/phase3_9b_geometry_idm/state_action/xy_only_high_weight/fold_phase3_9b_seed_392003/inverse_dynamics.pt` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `PASS` | `none` | No preflight issues found. |

## Scope

- This preflight does not run rollout.
- Phase3.10b runs trace audit only.
- No model training.
- No future DDPM training.
- No Phase4 or CPS is allowed.
