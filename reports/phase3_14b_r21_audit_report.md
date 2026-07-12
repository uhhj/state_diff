# Phase3.14b-r2.1 Validity and Ordered-Geometry Audit

- Verdict: `PASS` (diagnostic execution completed)
- Root cause: `phase314b_r21_validity_geometry_audit_completed`
- Device: `cpu`
- Validation rows: `102`
- Formal test read: `False`

## Reference calibration

| Reference | Original validity | Calibrated validity |
|---|---:|---:|
| `train_gt` | 0.823000 | 0.982000 |
| `fit_gt` | 0.839286 | 0.976190 |
| `calibration_gt` | 0.806452 | 0.987903 |
| `validation_gt` | 0.754902 | 0.970588 |
| `last_repeat_validation` | 0.980392 | 1.000000 |
| `deterministic_validation` | 0.000000 | 0.000000 |

## R2 checkpoints

| Config | Seed | Original valid | Calibrated valid | Ordered RMSE | Chamfer | Permutation gap |
|---|---:|---:|---:|---:|---:|---:|
| `epsilon_cosine_cap_0p5` | 31431 | 0.000000 | 0.000000 | 1.083886 | 0.971115 | 0.090566 |
| `epsilon_cosine_cap_0p5` | 31432 | 0.000000 | 0.000000 | 1.104039 | 0.995438 | 0.085530 |
| `epsilon_cosine_cap_0p5` | 31433 | 0.000000 | 0.000000 | 1.078199 | 0.959941 | 0.095735 |
| `sample_cosine` | 31431 | 0.000000 | 0.000000 | 0.118252 | 0.123884 | -0.006881 |
| `sample_cosine` | 31432 | 0.000000 | 0.000000 | 0.118019 | 0.123373 | -0.007089 |
| `sample_cosine` | 31433 | 0.000000 | 0.000000 | 0.118163 | 0.123375 | -0.006902 |
| `v_prediction_cosine` | 31431 | 0.000000 | 0.000000 | 0.173063 | 0.098999 | 0.068862 |
| `v_prediction_cosine` | 31432 | 0.000000 | 0.000000 | 0.173920 | 0.099823 | 0.069083 |
| `v_prediction_cosine` | 31433 | 0.000000 | 0.000000 | 0.174248 | 0.099854 | 0.068877 |

The separate analyzer assigns the scientific root cause.
No retraining, formal test read, IDM, execution, Phase4, or CPS was run.
