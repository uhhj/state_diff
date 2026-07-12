# Phase3.14b-r2.1 Final Report

- Verdict: `PASS`
- Root cause: `phase314b_r21_contract_miscalibration_and_ordered_geometry_failure_supported`
- Next step: `Phase3.14b-r2.2: freeze a train-only family-wise validity contract and add ordered-cable geometry repair to v_prediction_cosine; keep test and IDM blocked.`

| Evidence | Value |
|---|---:|
| `original_contract_miscalibrated` | `True` |
| `calibrated_contract_supported` | `True` |
| `ordered_geometry_failure` | `True` |
| `pointset_ordering_gap_supported` | `True` |
| `original_train_gt_validity` | `0.823` |
| `original_validation_gt_validity` | `0.7549019607843137` |
| `original_last_repeat_validity` | `0.9803921568627451` |
| `original_deterministic_validity` | `0.0` |
| `calibrated_validation_gt_validity` | `0.9705882352941176` |
| `calibrated_last_repeat_validity` | `1.0` |
| `calibrated_deterministic_validity` | `0.0` |
| `v_prediction_calibrated_validity_median` | `0.0` |
| `v_prediction_coordinate_validity_median` | `0.6924019607843137` |
| `v_prediction_segment_validity_median` | `0.0` |
| `v_prediction_ordered_rmse_median` | `0.1739195016413635` |
| `v_prediction_chamfer_median` | `0.09982332503697013` |
| `v_prediction_permutation_gap_median` | `0.06887689490766064` |
| `v_prediction_segment_stretch_p95_median` | `56.518205642700195` |
| `v_prediction_nearest_inversion_median` | `0.4052598394430236` |

- Formal test read: `False`
- DDPM retraining: `False`
- IDM: `False`
- Candidate action execution: `False`
- Phase4/CPS: `False`
