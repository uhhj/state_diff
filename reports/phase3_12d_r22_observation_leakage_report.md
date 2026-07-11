# Phase3.12d-r2.2 Observation-Space Leakage Audit

- Verdict: `FAIL`
- Root cause: `phase312d_r22_latent_condition_observably_leaked`
- Backend: `torch_linear_bce_lbfgs`
- Groups: `128 visible seeds`

## Horizon-20 Primary Comparison

| Feature | ROC-AUC | 95% CI | Balanced accuracy | Permutation p |
|---|---:|---|---:|---:|
| `xy` | `0.505157` | `[0.505463, 0.510407]` | `0.503906` | `0.000999` |
| `xy_velocity` | `0.891846` | `[0.849426, 0.928773]` | `0.808594` | `0.000999` |
| `full_state` | `0.888672` | `[0.846555, 0.926393]` | `0.789062` | `0.000999` |
| `model_x` | `0.879944` | `[0.837463, 0.919435]` | `0.777344` | `0.000999` |

## Controls

| Comparison | Max AUC | Max lower CI | Result |
|---|---:|---:|---|
| `free_a_vs_free_b` | `0.500000` | `0.500000` | `PASS` |
| `free_a_vs_hidden_unarmed` | `0.500000` | `0.500000` | `PASS` |
| `horizon0_free_a_vs_hidden_armed` | `0.500000` | `0.500000` | `PASS` |

## Interpretation

This diagnostic tests decodability for the specified seeds, horizons, and structured observation features. It does not prove permanent unobservability, CPS effectiveness, or model correctness.
