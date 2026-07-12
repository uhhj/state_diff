# Phase3.14b-r1 Final Report

- Verdict: `FAIL`
- Root cause: `phase314b_r1_scheduler_implementation_failed`
- Next step: `Repair the scheduler/step implementation and rerun this audit with the existing checkpoint before any retraining.`

| Evidence | Result |
|---|---:|
| `standardization_contract_pass` | `True` |
| `scheduler_true_epsilon_pass` | `True` |
| `manual_scheduler_equivalence_pass` | `False` |
| `terminal_amplification_large` | `True` |
| `high_timestep_error_dominates` | `True` |
| `partial_high_snr_better_than_terminal` | `True` |
| `posterior_noise_primary` | `False` |
| `ema_only_failure` | `False` |
| `low_timestep_denoising_failed` | `False` |

No retraining, IDM, candidate action execution, Phase4, or CPS was run.
