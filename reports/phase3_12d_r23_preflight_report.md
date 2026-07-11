# Phase3.12d-r2.3 Observation Contract Preflight

- Verdict: `WARN`
- Collection allowed: `True`
- Main HEAD: `6c4a886c2633016d8dcbb8130d75b8842b022fcd`
- Submodule: `e5525384af4af5bd0a02c3d1fd33ae84323d44e2`

## Checks

| Check | Result |
|---|---:|
| `main_branch_experiment1` | `True` |
| `submodule_clean` | `True` |
| `main_worktree_clean` | `False` |
| `import_numpy` | `True` |
| `import_torch` | `True` |
| `import_pybullet` | `True` |
| `import_ccda_phase3_data_io` | `True` |
| `import_ccda_phase3_rollout` | `True` |
| `import_ccda_phase3_observation_contract` | `True` |
| `import_phase3_12d_r22_integrity` | `True` |
| `import_phase3_12d_r22_collect_observation_leakage` | `True` |
| `import_phase3_12d_r22_analyze_observation_leakage` | `True` |
| `import_phase3_12d_r2_environment_audit` | `True` |
| `r22_summary_exists` | `True` |
| `r22_data_exists` | `True` |
| `windows_exists` | `True` |
| `r22_expected_root_cause` | `True` |
| `legacy_state_dim_135` | `True` |
| `proposed_state_dim_87` | `True` |
| `proposed_model_x_dim_303` | `True` |
| `r22_collector_reuse_api_available` | `True` |
| `r22_model_x_repeats_single_state` | `True` |
| `task_logs_pybullet_base_velocity` | `True` |
| `tensorflow_not_loaded` | `True` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `main_worktree_contains_preserved_changes` | ?? ccda_phase3/observation_contract.py
?? checkpoints/
?? reports/phase3_12_workers/
?? reports/phase3_12b_workers/
?? reports/phase3_12c_reset_workers/
?? reports/phase3_12c_rollout_workers/
?? reports/phase3_12d_candidate_effects.csv
?? reports/phase3_12d_candidate_oracle_report.md
?? reports/phase3_12d_candidate_oracle_summary.json
?? reports/phase3_12d_candidate_progress.json
?? reports/phase3_12d_no_phase4_confirmation.md
?? reports/phase3_12d_preflight_report.md
?? reports/phase3_12d_preflight_summary.json
?? reports/phase3_12d_query_headroom.csv
?? reports/phase3_12d_query_manifest.json
?? reports/phase3_12d_r23_preflight_report.md
?? reports/phase3_12d_r23_preflight_summary.json
?? reports/phase3_12d_r23_start.md
?? reports/phase3_12d_runtime_config.md
?? reports/phase3_12d_source_summary.csv
?? reports/phase3_12d_start.md
?? reports/phase3_12d_workers/
?? scripts/phase3_12d_analyze.py
?? scripts/phase3_12d_collect_queries.py
?? scripts/phase3_12d_common.py
?? scripts/phase3_12d_execute_candidates.py
?? scripts/phase3_12d_preflight.py
?? scripts/phase3_12d_r23_analyze_observation_contract.py
?? scripts/phase3_12d_r23_collect_trajectory_observations.py
?? scripts/phase3_12d_r23_preflight.py
?? scripts/phase3_12d_r23_run.sh
?? scripts/phase3_12d_run.sh
?? tests/test_phase3_12d_r23_observation_contract.py |
| `WARN` | `r22_model_x_is_not_real_causal_history` | pad_history([state], th) repeats one state |
| `WARN` | `simulator_bead_velocity_is_privileged_state` | bead velocity comes directly from PyBullet getBaseVelocity |

## Decision

- This phase does not change the physical task.
- It does not change the production training schema yet.
- It collects real no-action trajectories and separates privileged simulator velocity from observable position/proprio history.
- No candidate matrix, Phase4, or CPS is allowed.
