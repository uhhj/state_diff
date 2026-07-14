# Phase3.14b-r2.5.3 Reconstruction / Branch-Transport Gate Separation

## Verdict

- Verdict: `PASS` (train-only diagnostic chain completed)
- Root cause: `phase314b_r253_robot_proxy_reconstruction_conflation_supported`
- Supported mechanisms: `['exact_reconstruction_vs_branch_identity_separation', 'intermediate_horizon_fidelity_gap', 'exact_cable_fidelity_tail_gap', 'all_source_tail_failure', 'robot_proxy_reconstruction_conflation']`
- Next: `audit a separate robot-proxy trajectory-fidelity objective under the fixed cable branch-transport contract`
- Train-only recommendation: `None`
- Selected configuration: `None`

## Model gate separation

| Model | Exact reconstruction | Branch transport | One-step physical | Ordered topology | Cable geometry | Cable/robot error fraction | Failing sources |
|---|---:|---:|---:|---:|---:|---|---:|
| `v_only_frozen_control` | false | true | false | true | false | 0.08426/0.9157 | 10 |
| `ordered_mean_raw_g100` | false | true | true | true | true | 0.03034/0.9697 | 5 |
| `ordered_cvar_contract_g010` | false | true | true | true | true | 0.07222/0.9278 | 8 |

## Reconstruction components

| Model | z MSE | Ordered p95 | Segment p95 | Chain p95 | Failed components |
|---|---:|---:|---:|---:|---|
| `v_only_frozen_control` | 0.00583997 | 0.00498879 | 0.404936 | 0.0958517 | `['z_mse', 'segment_relative_error_p95']` |
| `ordered_mean_raw_g100` | 0.00512611 | 0.00215062 | 0.0813815 | 0.0186211 | `['z_mse']` |
| `ordered_cvar_contract_g010` | 0.00283722 | 0.00280444 | 0.093377 | 0.0247033 | `['z_mse']` |

## Per-horizon cable fidelity

| Model | h0 | h1 | h2 | h3 |
|---|---:|---:|---:|---:|
| `v_only_frozen_control` | false | false | true | true |
| `ordered_mean_raw_g100` | true | true | true | true |
| `ordered_cvar_contract_g010` | true | true | true | true |

## Synthetic controls

| Control | Exact reconstruction | Branch transport | Physical | Ordered topology | Cable geometry |
|---|---:|---:|---:|---:|---:|
| `bead_order_reversal` | false | false | true | false | false |
| `branch_swap` | false | false | true | true | false |
| `cable_translation` | false | false | true | true | false |
| `collapse_first_branch` | false | false | true | true | false |
| `exact_oracle` | true | true | true | true | true |
| `pair_mean` | false | false | false | true | false |
| `robot_proxy_offset` | false | true | true | true | true |

## Immutable reverse evidence

| Model | Candidate-quality gate | Validity | Valid-query | Both branches | Best-K RMSE |
|---|---:|---:|---:|---:|---:|
| `v_only_frozen_control` | false | 0.796875 | 1 | 0.75 | 0.00151318 |
| `ordered_mean_raw_g100` | true | 1 | 1 | 1 | 0.00114914 |
| `ordered_cvar_contract_g010` | true | 1 | 1 | 1 | 0.00124496 |

## Boundaries

- Historical reconstruction threshold changed: `False`
- Branch threshold changed: `False`
- Training changed: `False`
- Validation targets used: `False`
- Formal test read: `False`
- Formal training: `False`
- Candidate selected: `False`
- Checkpoint saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`

A PASS verdict means only that the gate-separation diagnostic chain completed.

## Resume3 finalizer-only corrections

- GPU pilot rerun: `False`
- Training rerun: `False`
- One-step rerun: `False`
- Reverse rerun: `False`
- Committed pilot SHA256: `a6a8aa2faf739e80159af7e70cb57b0e2a4c852ab5af5c2d7ecb29ff31116c86`
- Oracle producer key: `legacy_oracle_audit`
- Faulty consumer key: `exact_v_oracle_audit`
- Persisted oracle proof: `pipeline_controls.oracle_legacy_gate_pass`
- Parent-shell repository-root `PYTHONPATH`: `PASS`
- Script-local repository-root bootstrap: `PASS`
- Python executable: `/miniforge3/envs/coord_bimanual/bin/python3.9`
- Python: `3.9.15`
- NumPy: `1.23.3`
- PyTorch: `1.12.1.post200`
- CUDA runtime: `11.2`
- CUDA available during finalization: `False`
