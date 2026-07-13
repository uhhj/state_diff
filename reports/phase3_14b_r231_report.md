# Phase3.14b-r2.3.1 Tiny-Control Contract Correction

## Verdict

- Verdict: `PASS` (diagnostic correction completed, not model repair)
- Root cause: `phase314b_r231_random_noise_single_row_optimization_failure`
- Selected configuration: `None`
- Next: `debug timestep/noise coverage and denoiser conditioning on one train row`

## Corrected Contract

The historical r2.3 fixed-noise control trained on one fixed noisy tuple but
evaluated its memorization gate with newly sampled noise. The corrected gate
uses the exact training `x_t`, timestep and noise. Fresh-noise behavior is
reported separately.

Overfit gates are target-relative and do not require every training target to
pass the population physical-validity contract.

## Control Gates

- direct_one_row: `True`
- direct_unique_free_16: `True`
- fixed_one_row_v_only_exact_replay: `True`
- fixed_pairs_v_only_adamw_exact_replay: `True`
- fixed_pairs_v_only_adam_exact_replay: `True`
- fixed_pairs_v_only_wide_exact_replay: `True`
- fixed_pairs_r22_geometry_exact_replay: `True`
- random_one_row_t50: `False`
- random_unique_free_16_t50: `False`
- random_paired_16_t50_diagnostic_only: `False`

## Interpretation

- Historical r2.3 capacity conclusion invalidated:
  `True`
- Pair input max-absolute median:
  `0.0`
- Pair target ordered-RMSE median:
  `0.006412464560293967`

## Boundaries

- Validation target rows used by controls: `False`
- Formal test read: `False`
- Formal training: `False`
- Checkpoint saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`
