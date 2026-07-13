# Phase3.14b-r2.4 Train-Only Timestep-Conditioned Noisy-Skip Pilot

## Verdict

- Verdict: `PASS` (train-only pilot completed, not formal model repair)
- Root cause: `phase314b_r24_conditioned_multirow_generalization_failed`
- Train-only recommendation: `None`
- Selected configuration: `None`
- Next: `debug condition capacity and source-row batching on unique-free train rows`

## Stage gates

### one_row

- `analytic_x0_skip_residual_v`: `{"fresh_noise_all_t": true, "jvp_t50": true}`
- `analytic_x0_skip_residual_v_x0`: `{"fresh_noise_all_t": false, "jvp_t50": true}`
- `analytic_x0_skip_v`: `{"fresh_noise_all_t": false, "jvp_t50": true}`
- `learned_time_affine_v`: `{"fresh_noise_all_t": false, "jvp_t50": true}`

### paired_16


### unique_free_16

- `analytic_x0_skip_residual_v`: `{"fresh_noise_all_t": false, "jvp_t50": true}`

## Boundaries

- Validation targets used: `False`
- Formal test read: `False`
- Formal training: `False`
- Formal candidate selected: `False`
- Checkpoint saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`

The recommendation, when present, is train-only and authorizes only a separate train-only ordered-geometry pilot.
