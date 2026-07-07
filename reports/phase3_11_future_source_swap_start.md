# Phase3.11 Future-Source Swap / Branch Diagnosis Audit Start

- Timestamp: 2026-07-07T11:06:34+08:00
- Main branch: Experiment1
- Main HEAD: 8555def3a8a36e19d353b0dbf70f246a69755bf8
- origin/Experiment1: 8555def3a8a36e19d353b0dbf70f246a69755bf8

## Main status

```text
?? checkpoints/
?? reports/phase3_11_future_source_swap_start.md
?? scripts/phase3_11_analyze_future_source_swap.py
?? scripts/phase3_11_future_source_swap_preflight.py
?? scripts/phase3_11_future_source_swap_rollout.py
?? scripts/phase3_11_run_future_source_swap.sh
```

## Submodule

```text
 388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)
```

## Objective

Run a future-source swap audit after Phase3.10b found predicted future branch mismatch:
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- compare ddpm_mean, ddpm_best_of_k_by_train_nn, global_input_retrieval, condition_matched_retrieval
- condition_matched_retrieval is diagnostic upper bound only
