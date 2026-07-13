# Phase3.14b-r2.3.2 Single-Row Random-Noise Denoiser Isolation

## Verdict

- Verdict: `PASS` (diagnosis completed, not model repair)
- Root cause: `phase314b_r232_noisy_input_skip_path_deficiency_supported`
- Selected configuration: `None`
- Next: `design a train-only residual/noisy-skip denoiser pilot; formal validation remains blocked`

## Core checks

- Oracle v/x0 parity: `True`
- Timestep embedding audit: `True`
- Oracle v max abs: `2.1457672119140625e-06`
- Oracle x0 max abs: `4.76837158203125e-07`

## Control gates

- `cartesian_baseline`: `{"heldout_cartesian": false, "seen_cartesian": true}`
- `cartesian_residual`: `{"heldout_cartesian": false, "seen_cartesian": false}`
- `fixed_noise_all_t_baseline`: `{"seen_timestep_bank": true}`
- `fixed_noise_all_t_residual`: `{"seen_timestep_bank": true}`
- `fixed_t50_baseline`: `{"heldout_noise_bank": false, "seen_noise_bank": false}`
- `fixed_t50_residual`: `{"heldout_noise_bank": false, "seen_noise_bank": true}`
- `fixed_t50_time_affine`: `{"heldout_noise_bank": true, "seen_noise_bank": true}`
- `stream_baseline`: `{"heldout": false}`
- `stream_residual`: `{"heldout": false}`

## Boundaries

- Validation targets used: `False`
- Formal test read: `False`
- Formal training: `False`
- Candidate selected: `False`
- Checkpoint saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`

A PASS verdict means only that the train-only diagnostic evidence chain completed.
