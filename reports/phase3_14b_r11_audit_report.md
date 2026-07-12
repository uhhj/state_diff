# Phase3.14b-r1.1 Exact Scheduler Audit

- Verdict: `PASS` (diagnostic execution completed)
- Root cause: `phase314b_r11_exact_scheduler_audit_completed`
- Runtime device: `cpu`
- Diagnostic rows: `16`

| Gate | Value |
|---|---:|
| `standardization_roundtrip_max_abs` | `2.384185791015625e-07` |
| `scheduler_true_epsilon_x0_max_abs` | `0.0002416372299194336` |
| `exact_prev_allclose` | `True` |
| `exact_pred_x0_allclose` | `True` |
| `exact_prev_max_abs` | `0.0` |
| `exact_prev_max_relative_rms` | `0.0` |
| `legacy_prev_max_abs` | `4.553794860839844e-05` |
| `terminal_x0_error_amplification` | `2029.2114580785008` |
| `ema_high_bin_pred_x0_mse` | `15637.2998046875` |
| `ema_low_bin_pred_x0_mse` | `0.047346483916044235` |
| `ema_high_to_low_x0_mse_ratio` | `330273.72914144763` |
| `ema_partial_t10_chamfer` | `0.017152230022475123` |
| `ema_partial_t99_chamfer` | `21.53482037782669` |
| `ema_stochastic_final_chamfer` | `23.318753480911255` |
| `ema_posterior_mean_final_chamfer` | `23.118656396865845` |
| `raw_stochastic_final_chamfer` | `23.153687596321106` |
| `raw_posterior_mean_final_chamfer` | `22.853984236717224` |

The separate analyzer assigns the scientific root cause.
No retraining, IDM, candidate execution, Phase4, or CPS was run.
