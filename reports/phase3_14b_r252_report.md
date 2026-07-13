# Phase3.14b-r2.5.2 Paired One-Step / Reverse-Trajectory Attribution

## Verdict

- Verdict: `PASS` (train-only diagnostic chain completed)
- Root cause: `phase314b_r252_composite_one_step_gate_conflation_supported`
- Supported mechanisms: `['composite_gate_conflation', 'per_timestep_gradient_miscalibration']`
- Train-only recommendation: `None`
- Selected configuration: `None`

## One-step gate decomposition

| Variant | Denoising gate | Branch gate | Composite | Own closer | Separation p50 | Cosine p50 | Pipeline |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v_only_frozen_control` | false | true | false | 0.994792 | 0.97673 | 0.945718 | true |
| `ordered_mean_raw_g100` | false | true | false | 0.986979 | 1.01875 | 0.997607 | true |
| `ordered_cvar_contract_g010` | false | true | false | 0.997396 | 1.01017 | 0.990164 | true |

## Per-timestep branch transport and gradients

| Variant | t | Own closer | Separation | Cosine | Residual gain | Residual cosine | Grad med/p95 | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v_only_frozen_control` | 10 | 1 | 0.977741 | 0.979367 | 0.979381 | 0.999462 | 0/0 | true |
| `v_only_frozen_control` | 25 | 1 | 0.951386 | 0.934932 | 0.94372 | 0.998797 | 0/0 | true |
| `v_only_frozen_control` | 50 | 0.984375 | 0.990304 | 0.92482 | 0.940258 | 0.997745 | 0/0 | true |
| `ordered_mean_raw_g100` | 10 | 1 | 1.00121 | 0.997978 | 1.01569 | 0.999756 | 0.929478/0.954946 | true |
| `ordered_mean_raw_g100` | 25 | 1 | 1.02881 | 0.997532 | 1.00325 | 0.999486 | 1.85266/1.90041 | true |
| `ordered_mean_raw_g100` | 50 | 0.960938 | 1.03884 | 0.995886 | 0.982493 | 0.998703 | 4.00436/4.09085 | true |
| `ordered_cvar_contract_g010` | 10 | 1 | 1.00542 | 0.993913 | 0.980875 | 0.999597 | 0.0644212/0.067314 | true |
| `ordered_cvar_contract_g010` | 25 | 1 | 0.999475 | 0.992185 | 0.965889 | 0.999664 | 0.142817/0.153611 | true |
| `ordered_cvar_contract_g010` | 50 | 0.992188 | 1.02081 | 0.983417 | 0.976718 | 0.999146 | 0.255341/0.277336 | true |

## Predicted-x0 reverse trajectory

| Variant | Scheduler t | Validity | Valid-query | Best K | Inversion p95 | Both branches | Occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v_only_frozen_control` | 99 | 0.546875 | 0.75 | 0.00609891 | 0.0434783 | 0.125 | 0.75 |
| `v_only_frozen_control` | 90 | 0.609375 | 1 | 0.00183219 | 0.0434783 | 0.75 | 1 |
| `v_only_frozen_control` | 75 | 0.726562 | 1 | 0.0016356 | 0.0434783 | 0.875 | 1 |
| `v_only_frozen_control` | 50 | 0.75 | 1 | 0.00153687 | 0.115217 | 0.875 | 1 |
| `v_only_frozen_control` | 25 | 0.796875 | 1 | 0.00114337 | 0.0434783 | 0.875 | 1 |
| `v_only_frozen_control` | 10 | 0.78125 | 1 | 0.00110637 | 0.0717391 | 0.875 | 1 |
| `v_only_frozen_control` | 0 | 0.796875 | 1 | 0.00151317 | 0.28913 | 0.75 | 1 |
| `ordered_mean_raw_g100` | 99 | 0.578125 | 0.75 | 0.00701587 | 0.0434783 | 0.25 | 1 |
| `ordered_mean_raw_g100` | 90 | 0.78125 | 1 | 0.00124803 | 0.0434783 | 0.875 | 1 |
| `ordered_mean_raw_g100` | 75 | 0.929688 | 1 | 0.00134642 | 0.0434783 | 0.875 | 1 |
| `ordered_mean_raw_g100` | 50 | 0.960938 | 1 | 0.000630122 | 0.0434783 | 0.875 | 1 |
| `ordered_mean_raw_g100` | 25 | 1 | 1 | 0.000514939 | 0 | 0.875 | 1 |
| `ordered_mean_raw_g100` | 10 | 1 | 1 | 0.000577109 | 0 | 1 | 1 |
| `ordered_mean_raw_g100` | 0 | 1 | 1 | 0.00114914 | 0 | 1 | 1 |
| `ordered_cvar_contract_g010` | 99 | 0.554688 | 0.75 | 0.00598281 | 0.0434783 | 0.5 | 0.875 |
| `ordered_cvar_contract_g010` | 90 | 0.695312 | 1 | 0.00184386 | 0.0434783 | 0.875 | 1 |
| `ordered_cvar_contract_g010` | 75 | 0.914062 | 1 | 0.00118702 | 0.0434783 | 0.875 | 1 |
| `ordered_cvar_contract_g010` | 50 | 0.960938 | 1 | 0.000847014 | 0.0434783 | 0.875 | 1 |
| `ordered_cvar_contract_g010` | 25 | 0.992188 | 1 | 0.000645618 | 0.0434783 | 0.875 | 1 |
| `ordered_cvar_contract_g010` | 10 | 1 | 1 | 0.000582299 | 0.0434783 | 0.875 | 1 |
| `ordered_cvar_contract_g010` | 0 | 1 | 1 | 0.00124472 | 0 | 1 | 1 |

## Endpoint reproduction

| Variant | Validity | Valid-query | Best K | Inversion mean/p95/max | Both branches | Reproduces r2.5.1 |
|---|---:|---:|---:|---:|---:|---:|
| `v_only_frozen_control` | 0.796875 | 1 | 0.00151318 | 0.0465353/0.28913/0.391304 | 0.75 | true |
| `ordered_mean_raw_g100` | 1 | 1 | 0.00114914 | 0.0013587/0/0.0434783 | 1 | true |
| `ordered_cvar_contract_g010` | 1 | 1 | 0.00124496 | 0/0/0 | 1 | true |

## Boundaries

- Validation targets used: `False`
- Formal test read: `False`
- Formal training: `False`
- Candidate selected: `False`
- Checkpoint/model weights saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`

This audit attributes the one-step/reverse mismatch only. It does not repair ordered geometry.
