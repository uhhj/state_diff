# Phase3 PyTorch StateDiff CCDA Baseline Evaluation

## Scope

Phase3 evaluates contact-blind StateDiff-style baselines on `hidden-contact-cable-line` without hidden condition labels, pin ids, hidden contact metadata, success labels, contact concatenation, CPS guidance, or action feasibility classifiers.

## Runtime Backend

| Step | Python | Conda Env | Torch | Ravens |
|---|---|---|---|---|
| generate | not run in current report | NA | NA | NA |
| train_eval | `/miniforge3/envs/coord_bimanual/bin/python` | `coord_bimanual` | OK | not required |
| rollout | not run in current report | NA | NA | NA |
| aggregate | `/miniforge3/envs/coord_bimanual/bin/python` | `coord_bimanual` | OK | not required |

## Training Backend

| Backend | Count |
|---|---:|
| `torch` | 72 |

## Offline State Prediction

| Baseline | Condition | Wrong-Branch mean | Branch-Accuracy mean | Future Error mean | Averaging Score mean |
|---|---|---:|---:|---:|---:|
| paper_state | free | 0.372396 | 0.627604 | 0.516468 | 0.00090625 |
| paper_state | hidden_high_friction | NA | NA | 0.516261 | 0.00090625 |
| paper_state | hidden_pin | 0.627604 | 0.372396 | 0.533711 | 0.00090625 |
| state_action | free | 0.354167 | 0.645833 | 0.51585 | 0.00125573 |
| state_action | hidden_high_friction | NA | NA | 0.515737 | 0.00125573 |
| state_action | hidden_pin | 0.645833 | 0.354167 | 0.535677 | 0.00125573 |

## Inverse Dynamics

| Baseline | Condition | Action MSE mean | Action OOD mean |
|---|---|---:|---:|
| paper_state | free | 283.149 | 17.3679 |
| paper_state | hidden_high_friction | 283.149 | 17.3679 |
| paper_state | hidden_pin | 283.149 | 17.3679 |
| state_action | free | 283.087 | 17.3638 |
| state_action | hidden_high_friction | 283.087 | 17.3638 |
| state_action | hidden_pin | 283.087 | 17.3638 |

## Policy Execution

Policy rollout was not rerun for the current PyTorch train/eval report, so any older rollout CSV is not counted as current execution evidence.

## Phase3 Conclusion

This is a PyTorch smoke/medium validation, not the final full 5-fold x 3-seed result unless MODE=full was run.

The input consistency and leakage checks verify that paired `free` and `hidden_pin` samples have matched visible/proprio/action inputs, and probe classifiers cannot reliably recover hidden condition from the model inputs. Therefore, the branch ambiguity is not caused by accidental input leakage.

Across the current folds and random seeds, the contact-blind baselines exhibit elevated wrong-branch rate and/or branch ambiguity on the primary `free` vs `hidden_pin` CCDA subset. These results support moving to Phase4 only after a full-scale run if paper-level statistics are required.
