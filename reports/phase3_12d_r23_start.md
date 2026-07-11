# Phase3.12d-r2.3 Start

- Timestamp: `2026-07-11T12:39:36+08:00`
- Main branch: `Experiment1`
- Main HEAD: `6c4a886c2633016d8dcbb8130d75b8842b022fcd`
- Submodule: ` e5525384af4af5bd0a02c3d1fd33ae84323d44e2 external/deformable-ravens (heads/ccda-cable)`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`

## Objective

- Audit the observation contract after r2.2 found velocity-driven leakage.
- Collect real physics-step trajectories rather than repeating one state.
- Separate direct simulator bead velocity from causal XY history.
- Do not change the physical environment.
- Do not train StateDiff or IDM.
- Do not run candidate matrix, Phase4, or CPS.
