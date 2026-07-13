# Phase3.14b-r2.4.1 Conditioned Multirow and Source-Batching Audit

## Verdict

- Verdict: `PASS` (diagnosis completed, not model repair)
- Root cause: `phase314b_r241_condition_encoder_width_limit_supported`
- Train-only debug recommendation: `condition_width_1024`
- Selected configuration: `None`
- Next: `run an additive train-only wider condition-encoder pilot`

## Alignment and identifiability

- Source-row alignment: `True`
- Condition identifiability: `True`
- Exact duplicate conflict: `False`

## Direct condition controls

- `rows_16_width_1024`: `{"aggregate_gate": true, "all_source_gate": true, "ordered_rmse_p95": 1.7121384830770803e-08, "pass": true, "z_mse": 9.941908116729048e-15}`
- `rows_16_width_512`: `{"aggregate_gate": true, "all_source_gate": false, "ordered_rmse_p95": 0.0010958279513252193, "pass": false, "z_mse": 3.1331631627617615e-05}`
- `rows_2_width_512`: `{"aggregate_gate": true, "all_source_gate": true, "ordered_rmse_p95": 5.9434475527151016e-05, "pass": true, "z_mse": 1.6239707735076636e-07}`
- `rows_4_width_512`: `{"aggregate_gate": true, "all_source_gate": true, "ordered_rmse_p95": 1.7831675665990576e-08, "pass": true, "z_mse": 1.6132541236056714e-14}`
- `rows_8_width_512`: `{"aggregate_gate": true, "all_source_gate": true, "ordered_rmse_p95": 0.0003414539665122265, "pass": true, "z_mse": 6.995528078245797e-06}`

## Diffusion variants

- `balanced_anchor_prior`: `{"aggregate_gate": false, "all_source_gate": false, "condition_effect": true, "ordered_rmse_p95": 0.022585285593778807, "pass": false, "prior_drift_ratio": 1962.9399317012003, "sampling_relative_range": 0.0, "source_pass_fraction": 0.125, "z_mse": 0.015086232237990112}`
- `balanced_frozen_prior`: `{"aggregate_gate": true, "all_source_gate": true, "condition_effect": true, "ordered_rmse_p95": 0.000535458454624915, "pass": true, "prior_drift_ratio": 1.0, "sampling_relative_range": 0.0, "source_pass_fraction": 1.0, "z_mse": 6.100627014849927e-06}`
- `balanced_joint_fresh_optimizer`: `{"aggregate_gate": false, "all_source_gate": false, "condition_effect": true, "ordered_rmse_p95": 0.016841734431342934, "pass": false, "prior_drift_ratio": 562.596066937161, "sampling_relative_range": 0.0, "source_pass_fraction": 0.125, "z_mse": 0.008079643803808089}`
- `current_joint_random_reused_optimizer`: `{"aggregate_gate": false, "all_source_gate": false, "condition_effect": true, "ordered_rmse_p95": 0.06506623821266186, "pass": false, "prior_drift_ratio": 47707.999508363355, "sampling_relative_range": 0.013125, "source_pass_fraction": 0.125, "z_mse": 0.05982398581295029}`
- `random_anchor_prior`: `{"aggregate_gate": false, "all_source_gate": false, "condition_effect": true, "ordered_rmse_p95": 0.11149312599730297, "pass": false, "prior_drift_ratio": 8010.62005218745, "sampling_relative_range": 0.019125, "source_pass_fraction": 0.0, "z_mse": 0.11327179554791665}`

## Boundaries

- Validation targets used: `False`
- Formal test read: `False`
- Formal training: `False`
- Candidate selected: `False`
- Checkpoint saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`

Any recommendation is train-only and does not authorize formal validation.
