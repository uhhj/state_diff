# Phase3.12d-r2.4 Start Audit

- Timestamp (UTC): `2026-07-11T07:13:14.347095+00:00`
- Main branch: `Experiment1`
- Main HEAD: `0184eb65495f7cec2f3527f077003f808e7b0e94`
- Submodule branch: `ccda-cable`
- Submodule HEAD: `e5525384af4af5bd0a02c3d1fd33ae84323d44e2`
- Conda env: `coord_bimanual`
- Python: `3.9.15`

## Prior Result

- r2.3 root cause: `phase312d_r23_latent_condition_leaks_through_observable_motion`
- The rigid `hidden_breakaway_pin` remains historical diagnostic only.

## Scope

Implement and audit versioned `hidden_slack_breakaway_pin_v2` without modifying the old condition semantics. No training, candidate matrix, Phase4, or CPS is authorized.

## Existing Main Status

```text
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
?? reports/phase3_12d_runtime_config.md
?? reports/phase3_12d_source_summary.csv
?? reports/phase3_12d_start.md
?? reports/phase3_12d_workers/
?? scripts/phase3_12d_analyze.py
?? scripts/phase3_12d_collect_queries.py
?? scripts/phase3_12d_common.py
?? scripts/phase3_12d_execute_candidates.py
?? scripts/phase3_12d_preflight.py
?? scripts/phase3_12d_run.sh
```

The listed pre-existing untracked artifacts are preserved and are outside the r2.4 commit scope. The submodule was clean at audit start.
