# Phase3.14b-r1 Sampling-Scale Audit

- Verdict: `PASS` (diagnostic execution completed)
- Root cause: `phase314b_r1_sampling_scale_audit_completed`
- Diagnostic rows: `16`
- Device: `cpu`

| Diagnostic | Value |
|---|---:|
| `standardization_roundtrip_max_abs` | `2.384185791015625e-07` |
| `scheduler_true_epsilon_x0_max_abs` | `0.00024143606424331665` |
| `manual_vs_diffusers_prev_max_abs` | `6.914138793945312e-05` |
| `terminal_x0_error_amplification` | `2029.2114580785008` |
| `ema_high_bin_pred_x0_mse` | `44796.74609375` |
| `ema_low_bin_pred_x0_mse` | `0.07732386142015457` |
| `ema_high_to_low_x0_mse_ratio` | `579339.2268699305` |
| `ema_partial_t10_chamfer` | `0.01709555956767872` |
| `ema_partial_t99_chamfer` | `27.053202509880066` |
| `ema_stochastic_final_chamfer` | `19.348429083824158` |
| `ema_posterior_mean_final_chamfer` | `19.20256358385086` |
| `raw_stochastic_final_chamfer` | `20.027019500732422` |
| `raw_posterior_mean_final_chamfer` | `19.717155694961548` |

This report does not authorize retraining, IDM, candidate execution, Phase4, or CPS. The separate analyzer assigns the scientific root cause.
