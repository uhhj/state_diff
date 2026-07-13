# Phase3.14b-r2.5.1 Geometry-Gradient Calibration Audit

## Verdict

- Verdict: `PASS` (train-only diagnostic)
- Root cause: `phase314b_r251_paired_low_mid_geometry_transport_failed`
- Train-only recommendation: `None`
- Selected configuration: `None`

## Fixed contract

- Model: `frozen_p512_r512`
- Gradient calibration: actual residual-gradient norms
- Target ratios: `0.10 / 0.50 / 1.00`
- Reverse audit: `100 steps`, `K=16`

## Unique-free calibration

| Variant | PASS | Target | Multiplier | Observed median | Observed p95 | Prior drift | Ordered p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v_only_frozen_control` | true | 0 | 0 | 0 | 0 | 1 | 0.000158037 |
| `ordered_mean_raw_g010` | true | 0.1 | 5.09757e-05 | 0.286945 | 0.401196 | 1 | 0.000199453 |
| `ordered_mean_raw_g050` | true | 0.5 | 0.000254879 | 0.754133 | 1.57616 | 1 | 0.000130341 |
| `ordered_mean_raw_g100` | true | 1 | 0.000509757 | 1.05357 | 1.53318 | 1 | 0.000365664 |
| `ordered_cvar_raw_g010` | true | 0.1 | 4.88331e-05 | 0.280809 | 0.641401 | 1 | 0.000109894 |
| `ordered_cvar_raw_g050` | true | 0.5 | 0.000244165 | 0.595578 | 1.54519 | 1 | 0.00010729 |
| `ordered_cvar_raw_g100` | false | 1 | 0.000488331 | 5.69508 | 8.13671 | 1 | 8.19548e-05 |
| `ordered_cvar_contract_g010` | true | 0.1 | 4.88331e-05 | 0.226361 | 0.283919 | 1 | 0.000149667 |
| `ordered_cvar_contract_g050` | true | 0.5 | 0.000244165 | 0.354262 | 0.634888 | 1 | 0.000120203 |
| `ordered_cvar_contract_g100` | false | 1 | 0.000488331 | 5.69508 | 8.13671 | 1 | 8.19548e-05 |

## Paired and full reverse

| Variant | One-step | Validity | Valid-query | Best K | Inversion p95 | Both branches | Compare control |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v_only_frozen_control` | false | 0.796875 | 1 | 0.00151318 | 0.0465353/0.28913/0.391304 | 0.75 | true |
| `ordered_mean_raw_g010` | false | 0.992188 | 1 | 0.0015897 | 0.00407609/0.0434783/0.0434783 | 1 | true |
| `ordered_mean_raw_g050` | false | 0.78125 | 1 | 0.00177806 | 0.0064538/0.0434783/0.0869565 | 0.875 | false |
| `ordered_mean_raw_g100` | false | 1 | 1 | 0.00114914 | 0.0013587/0/0.0434783 | 1 | true |
| `ordered_cvar_raw_g010` | false | 0.945312 | 1 | 0.00181262 | 0.00543478/0.0434783/0.0869565 | 0.875 | false |
| `ordered_cvar_raw_g050` | false | 0.984375 | 1 | 0.00125315 | 0.00373641/0.0434783/0.0434783 | 0.75 | true |
| `ordered_cvar_contract_g010` | false | 1 | 1 | 0.00124496 | 0/0/0 | 1 | true |
| `ordered_cvar_contract_g050` | false | 0.9375 | 1 | 0.00165713 | 0.00815217/0.0434783/0.130435 | 0.875 | true |

## Boundaries

- Validation targets used: `False`
- Formal test read: `False`
- Formal training: `False`
- Candidate selected: `False`
- Checkpoint saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`

A PASS verdict means only that the train-only diagnostic chain completed.
