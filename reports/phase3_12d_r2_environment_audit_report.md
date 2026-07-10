# Phase3.12d-r2 Paired-Visible Latent Arming Audit

## Verdict

- Verdict: `FAIL`
- Root cause: `phase312d_r2_paired_visible_arming_or_causal_semantics_failed`
- Query-local snapshot allowed: `False`

## Visible parity / arming

| Seed | Pre max | Pre MAE | Immediate max | 20-step max | Anchor error | Legacy max |
|---:|---:|---:|---:|---:|---:|---:|
| 312000 | 0 | 0 | 0 | 4.1604042e-05 | 0 | 0.049968824 |
| 312001 | 0 | 0 | 0 | 4.8816204e-05 | 0 | 0.052405477 |
| 312002 | 0 | 0 | 0 | 0.00012534112 | 0 | 0.056391835 |
| 312003 | 0 | 0 | 0 | 0.00011876225 | 0 | 0.054614305 |
| 312500 | 0 | 0 | 0 | 0.00020721555 | 0 | 0.051805779 |
| 312501 | 0 | 0 | 0 | 0.00026060641 | 0 | 0.045477293 |
| 312502 | 0 | 0 | 0 | 0.00011014938 | 0 | 0.043527514 |
| 312503 | 0 | 0 | 0 | 0.00015699863 | 0 | 0.052883476 |

## Controlled causal probe

| Seed | Max trajectory divergence | Hidden released | Release step | Hook errors |
|---:|---:|---:|---:|---|
| 312000 | 0.064197 | True | 27 | None / None |
| 312500 | 0.053780 | True | 29 | None / None |

## Free goal-actionability probe

| Seed | Dense gain | Ordered gain | Fraction gain |
|---:|---:|---:|---:|
| 312000 | 0.074545 | 0.166528 | 0.500000 |
| 312500 | 0.273143 | 0.334978 | 0.166667 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `FAIL` | `short_horizon_post_arm_drift_failed` | {'seed': 312000, 'max_abs': 4.1604042053222656e-05, 'mae': 1.1082893858353296e-05, 'rmse': 1.5294088429817705e-05, 'fraction_diff': 0.0, 'curve_diff': 4.068643194394747e-06, 'breakaway_released': False, 'anchor_error': 0.0, 'hook_error': 'None'} |
| `FAIL` | `short_horizon_post_arm_drift_failed` | {'seed': 312002, 'max_abs': 0.00012534111738204956, 'mae': 2.2875067467490833e-05, 'rmse': 3.629826463284343e-05, 'fraction_diff': 0.0, 'curve_diff': 1.1610473761360443e-05, 'breakaway_released': False, 'anchor_error': 0.0, 'hook_error': 'None'} |
| `FAIL` | `short_horizon_post_arm_drift_failed` | {'seed': 312003, 'max_abs': 0.00011876225471496582, 'mae': 1.706741750240326e-05, 'rmse': 3.154580429289194e-05, 'fraction_diff': 0.0, 'curve_diff': 8.385041269287977e-06, 'breakaway_released': False, 'anchor_error': 0.0, 'hook_error': 'None'} |
| `FAIL` | `short_horizon_post_arm_drift_failed` | {'seed': 312500, 'max_abs': 0.0002072155475616455, 'mae': 3.467438121636709e-05, 'rmse': 6.108605172649337e-05, 'fraction_diff': 0.0, 'curve_diff': 1.1570746197547138e-06, 'breakaway_released': False, 'anchor_error': 0.0, 'hook_error': 'None'} |
| `FAIL` | `short_horizon_post_arm_drift_failed` | {'seed': 312501, 'max_abs': 0.00026060640811920166, 'mae': 3.801348308722178e-05, 'rmse': 6.659560154299594e-05, 'fraction_diff': 0.0, 'curve_diff': 5.582246249587197e-06, 'breakaway_released': False, 'anchor_error': 0.0, 'hook_error': 'None'} |
| `FAIL` | `short_horizon_post_arm_drift_failed` | {'seed': 312502, 'max_abs': 0.00011014938354492188, 'mae': 2.2931024432182312e-05, 'rmse': 3.261053761689311e-05, 'fraction_diff': 0.0, 'curve_diff': 1.1672636977596434e-06, 'breakaway_released': False, 'anchor_error': 0.0, 'hook_error': 'None'} |
| `FAIL` | `short_horizon_post_arm_drift_failed` | {'seed': 312503, 'max_abs': 0.0001569986343383789, 'mae': 2.734611431757609e-05, 'rmse': 4.957021687782606e-05, 'fraction_diff': 0.0, 'curve_diff': 3.903910991405121e-06, 'breakaway_released': False, 'anchor_error': 0.0, 'hook_error': 'None'} |

## Decision

- Any FAIL blocks all candidate smoke and full matrices.
- PASS means initial visible parity, zero-offset arming, short drift, causal divergence, breakaway release, and free actionability all hold.
- No training, Phase4, or CPS was run.
