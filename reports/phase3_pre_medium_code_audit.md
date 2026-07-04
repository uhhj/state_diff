# Phase3 Pre-Medium Code Audit

## Git Sync

```text
branch: Experiment1
head: 9ea0ed64685e4ed9f72620b7d84a1ee49ed82556
origin_experiment1: 9ea0ed64685e4ed9f72620b7d84a1ee49ed82556
status:
 M ccda_phase3/data_io.py
 M ccda_phase3/models.py
 M ccda_phase3/train_utils.py
 M reports/phase3_baseline_eval_predictions.csv
 M reports/phase3_baseline_eval_summary.json
 M reports/phase3_final_report.md
 M reports/phase3_fold_summary.csv
 M reports/phase3_prepare_windows_summary.json
 M reports/phase3_sanity_check_report.md
 M reports/phase3_sanity_check_summary.json
 M scripts/phase3_aggregate_folds.py
 M scripts/phase3_eval_baselines.py
 M scripts/phase3_prepare_windows.py
 M scripts/phase3_run_all.sh
 M scripts/phase3_sanity_check_metrics.py
 M scripts/phase3_train_baselines.py
?? reports/phase3_pre_medium_audit_report.md
?? reports/phase3_pre_medium_audit_summary.json
?? reports/phase3_pre_medium_code_audit.md
?? scripts/phase3_pre_medium_audit.py
```

## Submodule

```text
 b95180593d3366b754a4d8a4488d723bc9d61be5 external/deformable-ravens (heads/ccda-cable)
branch: ccda-cable
head: b95180593d3366b754a4d8a4488d723bc9d61be5
```

## coord_bimanual dependencies

```text
python /miniforge3/envs/coord_bimanual/bin/python
torch 1.12.1.post200 cuda False
diffusers 0.11.1
```

## Audit Notes

- `diffusers.DDPMScheduler` is available in `coord_bimanual`.
- Scheduler alignment uses `squaredcos_cap_v2`, `epsilon`, `fixed_small`, and `clip_sample=true`.
- Denoiser remains `mlp`; this is scheduler-aligned but not ConditionalUnet1D-identical.
- No medium/full/rollout was run for this audit.
