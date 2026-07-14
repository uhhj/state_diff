# Phase3.14b-r2.5.5 Stage B State-v3 Materialization

- Audit verdict: `PASS`
- Scientific status: `BLOCKED`
- Root cause: `phase314b_r255_stageb_state_v3_dataset_materialized_and_byte_reproducible`
- Required next path: `BUILD_NEW_STATE_V3_CACHE_AND_RERUN_ROBOT_PROXY_ATTRIBUTION`

## Materialized dataset

- Episodes: `896`
- State records: `5152`
- Action records/windows: `4256`
- State dimension: `67`
- Independent builds exact: `true`
- Dataset SHA256: `17a872d72dceadca1151821dbb5fd62982bf5ff1585852c05688b14d49d78b69`
- Windows SHA256: `0e55ae7e777536838e8ac70d33d830abc83b1940a42d97fd9b74081fce187319`

## Legacy invariants

- Cable history exact: `True`
- Cable future exact: `True`
- Action history exact: `True`
- Target action exact: `True`

## Boundary

- Legacy raw, windows, cache, reports, and submodule were not modified.
- No state-v3 training cache was built.
- No diffusion, reverse sampling, IDM, candidate execution, Phase4, or CPS was run.
- `robot_proxy_attribution_interpretable=false`.
- `train_only_recommendation=None` and `selected_configuration=None`.
