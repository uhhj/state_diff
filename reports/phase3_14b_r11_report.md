# Phase3.14b-r1.1 Final Report

- Verdict: `PASS`
- Root cause: `phase314b_r1_cosine_epsilon_terminal_snr_instability_supported`
- Next step: `Phase3.14b-r2 targeted objective/schedule repair using only MLP-DDPM paper_state before any full-matrix rerun.`

| Evidence | Result |
|---|---:|
| `standardization_contract_pass` | `True` |
| `true_epsilon_oracle_pass` | `True` |
| `exact_scheduler_equivalence_pass` | `True` |
| `terminal_amplification_large` | `True` |
| `high_timestep_x0_error_dominates` | `True` |
| `partial_t99_much_worse_than_t10` | `True` |
| `ema_and_raw_both_fail` | `True` |
| `posterior_noise_not_primary` | `True` |
| `legacy_algebraic_difference_nonzero` | `True` |

No DDPM retraining, IDM, candidate action execution, Phase4, or CPS was run.
