# Phase3 Condition Integration Start Audit

- Timestamp: `2026-07-05T14:59:22+08:00`
- Main branch: `Experiment1`
- Main HEAD: `315a03cc187994f0d6abd8592ced20712e7eacf2`
- origin/Experiment1: ``

## Main status

```text
M ccda_phase3/data_io.py
 M scripts/phase3_aggregate_folds.py
 M scripts/phase3_check_input_leakage.py
 M scripts/phase3_eval_baselines.py
 M scripts/phase3_generate_large_dataset.sh
 M scripts/phase3_pre_medium_audit.py
 M scripts/phase3_prepare_windows.py
 M scripts/phase3_run_all.sh
 M scripts/phase3_sanity_check_metrics.py
 M scripts/phase3_train_baselines.py
?? reports/phase3_condition_integration_start_audit.md
?? reports/phase3_generation_config.md
?? scripts/phase3_condition_integration_audit.py
```

## Submodule

```text
388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)
```

## Objective

Integrate Phase2.5c selected branch into Phase3:
- conditions: free, hidden_pin, hidden_high_friction, hidden_breakaway_pin
- primary pair: free vs hidden_breakaway_pin
- diagnostic pair: free vs hidden_pin
- leak-hardened matched-input audit
- no rollout / no Phase4 / no CPS
