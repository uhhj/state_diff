# Phase3.12d-r2.3 Observation Contract Audit

- Verdict: `FAIL`
- Root cause: `phase312d_r23_latent_condition_leaks_through_observable_motion`
- Backend: `torch_linear_bce_lbfgs`
- Visible seeds: `128`
- Trajectory rows: `10752`

## Decision comparison at step 20

| Feature | Legal | Family | Dim | ROC-AUC | 95% CI | Balanced acc | Paired acc | Sign p |
|---|---:|---|---:|---:|---|---:|---:|---:|
| `xy_single` | `True` | `position` | 48 | 0.515076 | [0.512695, 0.522339] | 0.507812 | 0.828125 | 2.20668e-14 |
| `sim_velocity_only` | `False` | `privileged_velocity` | 48 | 0.893311 | [0.850951, 0.930786] | 0.800781 | 0.851562 | 1.55583e-16 |
| `xy_sim_velocity` | `False` | `privileged_velocity` | 96 | 0.890289 | [0.847286, 0.928224] | 0.800781 | 0.843750 | 8.58862e-16 |
| `robot_only` | `True` | `robot_proprio` | 39 | 0.500000 | [0.500000, 0.500000] | 0.500000 | 0.500000 | 1 |
| `position_proprio_single` | `True` | `position_proprio` | 87 | 0.514282 | [0.512359, 0.521301] | 0.507812 | 0.835938 | 4.47573e-15 |
| `xy_history_s1` | `True` | `observable_motion` | 144 | 0.514893 | [0.512786, 0.522217] | 0.503906 | 0.828125 | 2.20668e-14 |
| `position_proprio_history_s1` | `True` | `observable_motion` | 261 | 0.516083 | [0.513123, 0.524231] | 0.507812 | 0.835938 | 4.47573e-15 |
| `causal_fd_history_s1` | `True` | `observable_motion` | 405 | 0.889465 | [0.845335, 0.928162] | 0.808594 | 0.835938 | 4.47573e-15 |
| `model_x_v2_s1` | `True` | `proposed_model_input` | 303 | 0.516998 | [0.514221, 0.525208] | 0.503906 | 0.835938 | 4.47573e-15 |
| `privileged_model_x_v1_s1` | `False` | `privileged_velocity` | 447 | 0.889130 | [0.844086, 0.927368] | 0.816406 | 0.835938 | 4.47573e-15 |
| `xy_history_s5` | `True` | `observable_motion` | 144 | 0.515900 | [0.513519, 0.523529] | 0.503906 | 0.843750 | 8.58862e-16 |
| `position_proprio_history_s5` | `True` | `observable_motion` | 261 | 0.514557 | [0.512634, 0.521667] | 0.500000 | 0.843750 | 8.58862e-16 |
| `causal_fd_history_s5` | `True` | `observable_motion` | 405 | 0.976807 | [0.958313, 0.990845] | 0.933594 | 0.992188 | 7.58194e-37 |
| `model_x_v2_s5` | `True` | `proposed_model_input` | 303 | 0.514557 | [0.512695, 0.521667] | 0.500000 | 0.843750 | 8.58862e-16 |
| `privileged_model_x_v1_s5` | `False` | `privileged_velocity` | 447 | 0.928680 | [0.892330, 0.960358] | 0.882812 | 0.914062 | 1.57568e-23 |
| `xy_history_s10` | `True` | `observable_motion` | 144 | 0.515717 | [0.513550, 0.523071] | 0.503906 | 0.843750 | 8.58862e-16 |
| `position_proprio_history_s10` | `True` | `observable_motion` | 261 | 0.514862 | [0.512817, 0.522156] | 0.507812 | 0.843750 | 8.58862e-16 |
| `causal_fd_history_s10` | `True` | `observable_motion` | 405 | 0.993164 | [0.981140, 1.000000] | 0.976562 | 1.000000 | 5.87747e-39 |
| `model_x_v2_s10` | `True` | `proposed_model_input` | 303 | 0.514862 | [0.512695, 0.522156] | 0.507812 | 0.843750 | 8.58862e-16 |
| `privileged_model_x_v1_s10` | `False` | `privileged_velocity` | 447 | 0.921356 | [0.886108, 0.953310] | 0.859375 | 0.937500 | 8.99211e-27 |
| `model_x_v2_s5_noise_00025m` | `True` | `noisy_visual_proxy` | 303 | 0.540253 | [0.490294, 0.591248] | 0.515625 | 0.546875 | 0.330936 |
| `model_x_v2_s5_noise_0005m` | `True` | `noisy_visual_proxy` | 303 | 0.551697 | [0.485596, 0.616699] | 0.507812 | 0.539062 | 0.426439 |
| `model_x_v2_s5_noise_001m` | `True` | `noisy_visual_proxy` | 303 | 0.469238 | [0.397215, 0.543581] | 0.503906 | 0.453125 | 0.330936 |

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
| `control_free_replicate` | `model_x_v2_s5_noise_00025m` | 0.465668 | 0.407349 | 0.375000 |
| `control_free_replicate` | `model_x_v2_s5_noise_0005m` | 0.534027 | 0.484985 | 0.492188 |
| `control_free_replicate` | `model_x_v2_s5_noise_001m` | 0.478638 | 0.408874 | 0.343750 |
| `control_hidden_unarmed` | `xy_single` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `sim_velocity_only` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `xy_sim_velocity` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `robot_only` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `position_proprio_single` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `xy_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `position_proprio_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `causal_fd_history_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `model_x_v2_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `privileged_model_x_v1_s1` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `xy_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `position_proprio_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `causal_fd_history_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `model_x_v2_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `privileged_model_x_v1_s5` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `xy_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `position_proprio_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `causal_fd_history_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `model_x_v2_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `privileged_model_x_v1_s10` | 0.500000 | 0.500000 | 0.500000 |
| `control_hidden_unarmed` | `model_x_v2_s5_noise_00025m` | 0.456146 | 0.412231 | 0.316406 |
| `control_hidden_unarmed` | `model_x_v2_s5_noise_0005m` | 0.516968 | 0.457733 | 0.453125 |
| `control_hidden_unarmed` | `model_x_v2_s5_noise_001m` | 0.509857 | 0.435933 | 0.421875 |
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
| `control_horizon0` | `model_x_v2_s5_noise_00025m` | 0.458405 | 0.412779 | 0.375000 |
| `control_horizon0` | `model_x_v2_s5_noise_0005m` | 0.448212 | 0.384674 | 0.359375 |
| `control_horizon0` | `model_x_v2_s5_noise_001m` | 0.522430 | 0.455199 | 0.453125 |

## Interpretation

- `sim_velocity_only` and privileged state features use direct PyBullet bead velocity and are not a visual observation contract.
- Causal histories use only previously captured bead XY and robot proprio; no simulator bead velocity is included.
- The proposed 87-D state-v2 schema is specified but not activated.
- No model training, candidate matrix, Phase4, or CPS was run.
