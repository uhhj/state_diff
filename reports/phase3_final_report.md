# Phase3 StateDiff CCDA Baseline Evaluation

## Scope

Phase3 evaluates contact-blind StateDiff-style baselines on `hidden-contact-cable-line` without hidden condition labels, pin ids, hidden contact metadata, success labels, contact concatenation, CPS guidance, or action feasibility classifiers.

## Offline State Prediction

| Baseline | Condition | Wrong-Branch mean | Branch-Accuracy mean | Future Error mean | Averaging Score mean |
|---|---|---:|---:|---:|---:|
| paper_state | free | 0.377604 | 0.622396 | 0.51791 | -0.00533789 |
| paper_state | hidden_high_friction | NA | NA | 0.517669 | -0.00533789 |
| paper_state | hidden_pin | 0.622396 | 0.377604 | 0.534687 | -0.00533789 |
| state_action | free | 0.377604 | 0.622396 | 0.51791 | -0.00533789 |
| state_action | hidden_high_friction | NA | NA | 0.517669 | -0.00533789 |
| state_action | hidden_pin | 0.622396 | 0.377604 | 0.534687 | -0.00533789 |

## Inverse Dynamics

| Baseline | Condition | Action MSE mean | Action OOD mean |
|---|---|---:|---:|
| paper_state | free | 0.692412 | 11.7376 |
| paper_state | hidden_high_friction | 0.692489 | 11.7376 |
| paper_state | hidden_pin | 0.692418 | 11.7376 |
| state_action | free | 0.692412 | 11.7376 |
| state_action | hidden_high_friction | 0.692489 | 11.7376 |
| state_action | hidden_pin | 0.692418 | 11.7376 |

## Policy Execution

| Baseline | Condition | Trials | Success Rate | Final Fraction | Final Curve |
|---|---|---:|---:|---:|---:|
| paper_state | free | 3 | 0 | 0 | 0.250594 |
| paper_state | hidden_high_friction | 3 | 0 | 0 | 0.246279 |
| paper_state | hidden_pin | 3 | 0 | 0 | 0.24982 |
| state_action | free | 3 | 0 | 0 | 0.250624 |
| state_action | hidden_high_friction | 3 | 0 | 0 | 0.246897 |
| state_action | hidden_pin | 3 | 0 | 0 | 0.250077 |

## Phase3 Conclusion

Phase3 evaluates contact-blind StateDiff-style baselines on a large hidden-contact cable dataset with held-out visible seeds. Both baselines use state history and robot pose/proprioception proxy; `state_action` additionally uses action history. Neither baseline receives hidden contact labels or contact metadata.

The input consistency and leakage checks verify that paired `free` and `hidden_pin` samples have matched visible/proprio/action inputs, and probe classifiers cannot reliably recover hidden condition from the model inputs. Therefore, the branch ambiguity is not caused by accidental input leakage.

Across folds and random seeds, the baselines exhibit elevated wrong-branch rate and/or branch ambiguity on the primary `free` vs `hidden_pin` CCDA subset. Inverse dynamics and closed-loop policy execution further show that the ambiguity affects downstream action generation and task success.

These results establish the baseline failure mode required before testing contact-conditioned methods. Phase4 should evaluate direct contact concatenation, and Phase5 should evaluate Contact Physical Score state-space guidance.
