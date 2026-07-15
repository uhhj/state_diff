# Phase3.14b-r2.5.6 Stage A Cable-only / IDM Contract

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r256_stagea_cable_only_contract_supported_paired_action_intervention_missing`
- Required next path: `RUN_CABLE_ONLY_DIFFUSION_DIAGNOSTIC_AND_COLLECT_ACTION_DIVERSE_IDM_DATA`

## Cable-only diffusion contract

- Condition: `state-v3 history + past action history`
- Condition dimension: `243`
- Target: `ordered cable XY future only`
- Target shape: `[4,48]`
- Robot history retained as condition: `true`
- Robot future in target: `false`

## Deployable IDM audit

- Train full-horizon rows: `1000`
- Paired episode groups: `192`
- Best cable feature: `history_plus_next_cable`
- Cable incremental gain: `0.509411142`
- Maximum permuted gain: `0.56632068`
- Gain over permuted: `-0.0569095381`
- Active action dimensions: `4`
- Unique action vectors: `500`
- Paired action-diverse fraction: `0`
- Formal IDM data ready: `false`

## Boundary

- Historical state-v2/v3 cache and evidence were not modified.
- Historical full-state diffusion and IDM fields were not reused.
- No diffusion training, reverse sampling, formal IDM training, candidate execution, Phase4, or CPS was run.
- `train_only_recommendation=None` and `selected_configuration=None`.
