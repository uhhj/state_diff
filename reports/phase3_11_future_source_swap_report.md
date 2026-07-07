# Phase3.11 Future-Source Swap / Branch Diagnosis Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase311_future_source_swap_inconclusive_or_not_supported`
- Episode rows: `36`
- Step rows: `576`
- Progress status: `completed`

## Per-Condition Future Source Comparison

| Condition | DDPM mean | DDPM best-k | Global retrieval | Condition retrieval | Best-k Δ | Global Δ | Condition Δ | DDPM match | Best-k match | Condition match | DDPM pull | Condition pull |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free` | 0.0278 | 0.0139 | 0.0556 | 0.0000 | -0.0139 | 0.0278 | -0.0278 | 0.000 | 0.167 | 1.000 | 0.0620 | 0.0816 |
| `hidden_breakaway_pin` | 0.0833 | 0.0556 | 0.0694 | 0.0000 | -0.0278 | -0.0139 | -0.0833 | 0.208 | 0.271 | 1.000 | 0.1157 | 0.1657 |
| `hidden_high_friction` | 0.2222 | 0.0972 | 0.1806 | 0.0000 | -0.1250 | -0.0417 | -0.2222 | 0.792 | 0.542 | 1.000 | 0.1282 | 0.1224 |

## Episode Summary

| IDM | Future source | Condition | Rows | OK | Timeout | Success | Final fraction | Future match | Future NN L2 | Pull len | Action OOD | Uses condition label | Failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `phase39b_xy_only_high_weight` | `condition_matched_retrieval` | `free` | 3 | 3 | 0 | 0.000 | 0.0000 | 1.000 | 0.0000 | 0.0816 | 0.424 | 1 | 0 |
| `phase39b_xy_only_high_weight` | `condition_matched_retrieval` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.0000 | 1.000 | 0.0000 | 0.1657 | 0.563 | 1 | 0 |
| `phase39b_xy_only_high_weight` | `condition_matched_retrieval` | `hidden_high_friction` | 3 | 3 | 0 | 0.000 | 0.0000 | 1.000 | 0.0000 | 0.1224 | 0.398 | 1 | 0 |
| `phase39b_xy_only_high_weight` | `ddpm_best_of_k_by_train_nn` | `free` | 3 | 3 | 0 | 0.000 | 0.0139 | 0.167 | 0.0688 | 0.1988 | 0.459 | 0 | 0 |
| `phase39b_xy_only_high_weight` | `ddpm_best_of_k_by_train_nn` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.0556 | 0.271 | 0.0689 | 0.1072 | 0.338 | 0 | 0 |
| `phase39b_xy_only_high_weight` | `ddpm_best_of_k_by_train_nn` | `hidden_high_friction` | 3 | 3 | 0 | 0.000 | 0.0972 | 0.542 | 0.0696 | 0.1092 | 0.372 | 0 | 0 |
| `phase39b_xy_only_high_weight` | `ddpm_mean` | `free` | 3 | 3 | 0 | 0.000 | 0.0278 | 0.000 | 0.0475 | 0.0620 | 0.190 | 0 | 0 |
| `phase39b_xy_only_high_weight` | `ddpm_mean` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.0833 | 0.208 | 0.0467 | 0.1157 | 0.339 | 0 | 0 |
| `phase39b_xy_only_high_weight` | `ddpm_mean` | `hidden_high_friction` | 3 | 3 | 0 | 0.000 | 0.2222 | 0.792 | 0.0554 | 0.1282 | 0.321 | 0 | 0 |
| `phase39b_xy_only_high_weight` | `global_input_retrieval` | `free` | 3 | 3 | 0 | 0.000 | 0.0556 | 1.000 | 0.0000 | 0.0817 | 0.429 | 0 | 0 |
| `phase39b_xy_only_high_weight` | `global_input_retrieval` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.0694 | 0.000 | 0.0000 | 0.0689 | 0.489 | 0 | 0 |
| `phase39b_xy_only_high_weight` | `global_input_retrieval` | `hidden_high_friction` | 3 | 3 | 0 | 0.000 | 0.1806 | 0.000 | 0.0000 | 0.0954 | 0.370 | 0 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `condition_matched_future_restores_branch_match` | {'ddpm_match': 0.20833333333333334, 'condition_match': 1.0} |
| `WARN` | `future_source_swap_not_clearly_supported` | {'condition': 'hidden_breakaway_pin', 'ddpm_mean_final': 0.08333333333333333, 'ddpm_bestk_final': 0.05555555555555555, 'global_retrieval_final': 0.06944444444444445, 'condition_retrieval_final': 0.0, 'bestk_improvement': -0.027777777777777776, 'global_retrieval_improvement': -0.013888888888888881, 'condition_retrieval_improvement': -0.08333333333333333, 'ddpm_mean_match': 0.20833333333333334, 'ddpm_bestk_match': 0.2708333333333333, 'global_retrieval_match': 0.0, 'condition_retrieval_match': 1.0, 'ddpm_mean_pull': 0.11568248774468277, 'ddpm_bestk_pull': 0.10715695101922999, 'global_retrieval_pull': 0.06893883649415027, 'condition_retrieval_pull': 0.16566480183973908} |

## Interpretation

- This is a future-source swap diagnostic only.
- `condition_matched_retrieval` uses condition labels and is a diagnostic upper bound, not deployable policy evidence.
- No model training was run.
- No future DDPM was trained.
- No Phase4 or CPS was run.
- If condition-matched retrieval beats DDPM mean, future branch conditioning is the likely blocker.
- If best-of-k beats mean, DDPM sampling contains useful branch information but mean aggregation is harmful.
