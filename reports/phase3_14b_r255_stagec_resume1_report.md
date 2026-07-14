# Phase3.14b-r2.5.5 Stage C State-v3 Cache and Attribution

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r255_stagec_robot_future_not_deployably_predictable_or_idm_material`
- Required next path: `REMOVE_ROBOT_FUTURE_FROM_DIFFUSION_TARGET_AND_REDEFINE_IDM_INPUT`

## Immutable cache

- Cache rows: `4256`
- Train rows: `2440`
- Paired window keys: `2128`
- Cache SHA256: `c342284cfcfcadb9a1412cf3ec93c27887eb2560bf8be4bf4e6936f037797c17`
- Train-only view SHA256: `18bf6148f9e8041c6e0d937c750d3fc750f87ae371be750d6956aa7c59558b73`
- Two independent cache builds exact: `true`
- Promotion exact: `true`

## Train-only robot-proxy attribution

- Attribution workers exact: `true`
- State-v3 schema contract: `true`
- Best deployable predictor: `cable_history_ridge`
- Best deployable NMSE: `1.09229393`
- Action-conditioned gain: `-491303.247`
- Robot future action-materiality gain: `0.643173659`

## Boundary

- The legacy state-v2 cache and Stage-B artifacts were not modified.
- Attribution used only the train-only view and entire paired episode groups stayed in one fold.
- No diffusion model or historical state-v2 prediction was used.
- No reverse sampling, formal IDM training, candidate execution, Phase4, or CPS was run.
- `train_only_recommendation=None` and `selected_configuration=None`.
