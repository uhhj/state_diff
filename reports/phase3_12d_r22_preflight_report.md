# Phase3.12d-r2.2 Preflight

- Verdict: `WARN`
- Collection allowed: `True`
- Classifier backend: `torch_1.12.1.post200_linear_bce`

## Checks

| Check | Result |
|---|---:|
| `main_branch_experiment1` | `True` |
| `submodule_clean` | `True` |
| `tensorflow_not_loaded` | `True` |
| `dedicated_tests_pass` | `True` |
| `shared_state_safety_helper` | `True` |
| `strict_visible_seed_parser` | `True` |
| `strict_json_rejects_nonfinite` | `True` |
| `windows_exist` | `True` |
| `action_codec_exists` | `True` |
| `state_checkpoint_exists` | `True` |
| `idm_checkpoint_report_exists` | `True` |
| `r21_report_exists` | `True` |
| `train_raw_source_exists` | `True` |
| `heldout_raw_source_exists` | `True` |
| `r21_root_cause_expected` | `True` |
| `torch_linear_fallback_available` | `True` |
| `no_phase4_or_cps_scope` | `True` |

## Warnings

- `main_worktree_contains_preserved_local_changes`: M ccda_phase3/data_io.py;  M ccda_phase3/rollout.py; ?? checkpoints/; ?? reports/phase3_12_workers/; ?? reports/phase3_12b_workers/; ?? reports/phase3_12c_reset_workers/; ?? reports/phase3_12c_rollout_workers/; ?? reports/phase3_12d_candidate_effects.csv; ?? reports/phase3_12d_candidate_oracle_report.md; ?? reports/phase3_12d_candidate_oracle_summary.json; ?? reports/phase3_12d_candidate_progress.json; ?? reports/phase3_12d_no_phase4_confirmation.md; ?? reports/phase3_12d_preflight_report.md; ?? reports/phase3_12d_preflight_summary.json; ?? reports/phase3_12d_query_headroom.csv; ?? reports/phase3_12d_query_manifest.json; ?? reports/phase3_12d_r22_bootstrap_metrics.csv; ?? reports/phase3_12d_r22_control_summary.csv; ?? reports/phase3_12d_r22_dataset_distribution.csv; ?? reports/phase3_12d_r22_distance_summary.csv; ?? reports/phase3_12d_r22_duplicate_windows.csv; ?? reports/phase3_12d_r22_feature_health.csv; ?? reports/phase3_12d_r22_fold_metrics.csv; ?? reports/phase3_12d_r22_no_phase4_confirmation.md; ?? reports/phase3_12d_r22_observation_leakage_report.md; ?? reports/phase3_12d_r22_observation_leakage_summary.json; ?? reports/phase3_12d_r22_predictions.csv; ?? reports/phase3_12d_r22_preflight_report.md; ?? reports/phase3_12d_r22_preflight_summary.json; ?? reports/phase3_12d_r22_repo_data_audit_report.md; ?? reports/phase3_12d_r22_repo_data_audit_summary.json; ?? reports/phase3_12d_r22_repo_discovery.json; ?? reports/phase3_12d_r22_source_reconstruction.csv; ?? reports/phase3_12d_r22_split_overlap.csv; ?? reports/phase3_12d_r22_start.md; ?? reports/phase3_12d_runtime_config.md; ?? reports/phase3_12d_source_summary.csv; ?? reports/phase3_12d_start.md; ?? reports/phase3_12d_workers/; ?? scripts/phase3_12d_analyze.py; ?? scripts/phase3_12d_collect_queries.py; ?? scripts/phase3_12d_common.py; ?? scripts/phase3_12d_execute_candidates.py; ?? scripts/phase3_12d_preflight.py; ?? scripts/phase3_12d_r22_analyze_observation_leakage.py; ?? scripts/phase3_12d_r22_collect_observation_leakage.py; ?? scripts/phase3_12d_r22_integrity.py; ?? scripts/phase3_12d_r22_preflight.py; ?? scripts/phase3_12d_r22_repo_data_audit.py; ?? scripts/phase3_12d_r22_run.sh; ?? scripts/phase3_12d_run.sh; ?? tests/test_ccda_phase3_state_safety.py; ?? tests/test_ccda_phase3_visible_seed.py; ?? tests/test_phase3_12d_r22_integrity.py
- `legacy_environment_semantics`: existing windows/checkpoints are a bridge diagnostic only
