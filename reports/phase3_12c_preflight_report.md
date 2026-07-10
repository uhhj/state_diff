# Phase3.12c Matched-Reset Preflight

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
| `import_ccda_phase3_metrics` | `True` |
| `import_ccda_phase3_train_utils` | `True` |
| `import_phase3_policy_rollout` | `True` |
| `import_phase3_12b_proxy_score_rollout` | `True` |
| `import_phase3_12c_matched_reset_common` | `True` |
| `forbidden_modules_not_loaded` | `True` |
| `task_registered` | `True` |
| `task_instantiates` | `True` |
| `task_has_required_conditions` | `True` |
| `task_has_primary` | `True` |
| `task_has_settle_attribute` | `True` |
| `windows_loadable` | `True` |
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
| `phase312b_summary_exists` | `True` |
| `phase312b_completed` | `True` |
| `phase312b_no_phase4_scope` | `True` |
| `running_in_coord_bimanual` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `phase312b_selector_dependent_pair_group_present` | True |
| `WARN` | `phase312b_explicit_rng_seeding_incomplete` | {'random.seed': False, 'np.random.seed': False, 'torch.manual_seed': False} |
| `WARN` | `phase312b_primary_comparison_uses_final_fraction` | True |

## Scope

- Preflight only.
- No rollout.
- No model training.
- No future DDPM training.
- No Phase4.
- No CPS.

## Important

- WARN findings about Phase3.12b are expected diagnostic findings and do not block Phase3.12c.
- Any FAIL finding blocks reset audit and paired selector evaluation.
