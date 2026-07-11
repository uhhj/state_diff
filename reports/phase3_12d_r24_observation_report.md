# Phase3.12d-r2.4 No-Action Observation Audit

- Verdict: `PASS`
- Root cause: `phase312d_r24_no_action_observation_hiddenness_passed`
- Backend: `torch_linear_bce_lbfgs`
- Visible seeds: `128`
- Trajectory rows: `123392`

## Decision comparison at step 240

| Feature | Legal | Family | Dim | ROC-AUC | 95% CI | Balanced acc | Paired acc | Sign p |
|---|---:|---|---:|---:|---|---:|---:|---:|
| `xy_single` | `True` | `position` | 48 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `sim_velocity_only` | `False` | `privileged_velocity` | 48 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `xy_sim_velocity` | `False` | `privileged_velocity` | 96 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `robot_only` | `True` | `robot_proprio` | 39 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `position_proprio_single` | `True` | `position_proprio` | 87 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `xy_history_s1` | `True` | `observable_motion` | 144 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `position_proprio_history_s1` | `True` | `observable_motion` | 261 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `causal_fd_history_s1` | `True` | `observable_motion` | 405 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `model_x_v2_s1` | `True` | `proposed_model_input` | 303 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `privileged_model_x_v1_s1` | `False` | `privileged_velocity` | 447 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `xy_history_s5` | `True` | `observable_motion` | 144 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `position_proprio_history_s5` | `True` | `observable_motion` | 261 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `causal_fd_history_s5` | `True` | `observable_motion` | 405 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `model_x_v2_s5` | `True` | `proposed_model_input` | 303 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `privileged_model_x_v1_s5` | `False` | `privileged_velocity` | 447 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `xy_history_s10` | `True` | `observable_motion` | 144 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `position_proprio_history_s10` | `True` | `observable_motion` | 261 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `causal_fd_history_s10` | `True` | `observable_motion` | 405 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `model_x_v2_s10` | `True` | `proposed_model_input` | 303 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `privileged_model_x_v1_s10` | `False` | `privileged_velocity` | 447 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |

## Controls

| Comparison | Feature | AUC | AUC lower CI | Paired lower CI |
|---|---|---:|---:|---:|
| `control_free_replicate` | `xy_single` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `sim_velocity_only` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `xy_sim_velocity` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `robot_only` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `position_proprio_single` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `xy_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `position_proprio_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `causal_fd_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `model_x_v2_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `privileged_model_x_v1_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `xy_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `position_proprio_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `causal_fd_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `model_x_v2_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `privileged_model_x_v1_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `xy_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `position_proprio_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `causal_fd_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `model_x_v2_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_free_replicate` | `privileged_model_x_v1_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `xy_single` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `sim_velocity_only` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `xy_sim_velocity` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `robot_only` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `position_proprio_single` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `xy_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `position_proprio_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `causal_fd_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `model_x_v2_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `privileged_model_x_v1_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `xy_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `position_proprio_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `causal_fd_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `model_x_v2_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `privileged_model_x_v1_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `xy_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `position_proprio_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `causal_fd_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `model_x_v2_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_v2_replicate` | `privileged_model_x_v1_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `xy_single` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `sim_velocity_only` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `xy_sim_velocity` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `robot_only` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `position_proprio_single` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `xy_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `position_proprio_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `causal_fd_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `model_x_v2_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `privileged_model_x_v1_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `xy_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `position_proprio_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `causal_fd_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `model_x_v2_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `privileged_model_x_v1_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `xy_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `position_proprio_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `causal_fd_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `model_x_v2_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_horizon0` | `privileged_model_x_v1_s10` | 0.500000 | 0.500000 | 0.500000 |

## Interpretation

- `sim_velocity_only` and privileged state features use direct PyBullet bead velocity and are not a visual observation contract.
- Causal histories use only previously captured bead XY and robot proprio; no simulator bead velocity is included.
- The proposed 87-D state-v2 schema is specified but not activated.
- No model training, candidate matrix, Phase4, or CPS was run.
