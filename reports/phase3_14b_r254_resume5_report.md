# Phase3.14b-r2.5.4 Resume5 Determinism Audit

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r254_resume5_same_device_residual_determinism_and_cable_functional_nonregression_supported`
- Scoped static tests: `438 passed`
- Same-device residual determinism: `true`
- Exact match to the prior RTX-4090 Resume3 run: `true`
- Cable functional non-regression: `true`
- Functional reproduction contract: `true`
- Historical Boolean exact agreement: `false`
- Robot-proxy attribution interpretable: `false`

## Contract design

Same-device identity is exact SHA equality over initialization, optimizer, RNG/sampler streams, histories, final residual state, and prediction. Cross-device functional reproduction introduces no new numeric tolerance: it requires all fixed cable gates to pass and forbids any historical true-to-false regression. Full-state and robot-proxy metrics remain excluded because the pinned logger schema is defective.

## Per-model result

| Model | repeat exact | Resume3 exact | cable non-regression | historical Boolean exact |
|---|---:|---:|---:|---:|
| `v_only_frozen_control` | true | true | true | false |
| `ordered_mean_raw_g100` | true | true | true | true |
| `ordered_cvar_contract_g010` | true | true | true | true |

## Boundary

The legacy loader eagerly materializes the locked NPZ, but this audit never indexes validation/formal target rows and never uses them in training or metrics. No reverse sampling, IDM, candidate execution, Phase4, or CPS was used. No checkpoint, weight tensor, prediction tensor, RNG state, or optimizer state was persisted. Robot-proxy metrics remain uninterpretable and no configuration is selected.

Next: `Phase3.14b-r2.5.5 robot-proxy schema repair, provenance migration, and train-cache regeneration proposal`.
