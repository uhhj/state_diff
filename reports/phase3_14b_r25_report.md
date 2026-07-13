# Phase3.14b-r2.5 Frozen-Prior Ordered-Geometry Pilot

## Verdict

- Verdict: `PASS` (train-only diagnostic; not formal model repair)
- Root cause: `phase314b_r25_geometry_gradient_scaling_failed`
- Train-only recommendation: `None`
- Selected configuration: `None`

## Fixed model contract

- Architecture: `factorized_analytic_x0_skip_mlp`
- Prior/residual widths: `512 / 512`
- Prior policy: `strict frozen after warmup`
- Objective: `v_prediction` with optional train-only ordered-geometry loss
- Reverse audit: `100 official scheduler steps`, `K=16`

## Unique-free controls

| Variant | PASS | Gradient ratio | Prior drift | z MSE | Ordered p95 | Validity |
|---|---:|---:|---:|---:|---:|---:|
| `ordered_cvar_contract` | false | 28.8141 | 1 | 2.09392e-07 | 7.01825e-05 | 1 |
| `ordered_cvar_raw` | false | 67.4221 | 1 | 2.35759e-07 | 7.67717e-05 | 1 |
| `ordered_mean_raw` | false | 30.1933 | 1 | 4.12785e-07 | 9.19455e-05 | 1 |
| `v_only_frozen_control` | true | 0 | 1 | 7.22067e-07 | 0.000158037 | 1 |

## Paired low/mid and full-reverse controls

| Variant | One-step | Reverse validity | Valid-query | Both branches | Compare control |
|---|---:|---:|---:|---:|---:|
| `v_only_frozen_control` | false | 0.796875 | 1 | 0.75 | true |

## Boundaries

- Validation targets used: `False`
- Formal test read: `False`
- Formal training: `False`
- Candidate selected: `False`
- Checkpoint saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`

A PASS verdict means only that the train-only diagnostic evidence chain completed.
