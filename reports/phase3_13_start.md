# Phase3.13 Start

- Timestamp (UTC): `2026-07-11T11:44:47.034343+00:00`
- Main HEAD: `b9721c5332018a2699841a810397683a0e6930a3`
- Submodule HEAD: `d30d4862fdf88c85f2c13fe01376f2a8e28f0303`
- r2.4 gate: `phase312d_r24_slack_breakaway_v2_environment_supported`
- Legacy tracked candidates: `1634`
- Local data/checkpoint/report files: `3272`
- Local bytes inventoried: `396583422`

## Scope

Migrate the formal task and observation contract to state-v2, validate a new paired dataset, then remove deprecated CCDA assets from the branch tip without rewriting Git history. No model training, candidate matrix, Phase4, or CPS is authorized.

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
