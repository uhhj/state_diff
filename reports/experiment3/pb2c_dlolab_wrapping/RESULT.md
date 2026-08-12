# PB2-C REV1 DLO-Lab Wrapping Natural Pair Discovery

Verdict: `PB2C_NATURAL_WINDING_PAIRS_FOUND`

## Rollout validity

- Total rollouts: 128
- Official-valid rollouts: 123
- Official-invalid rollouts: 5
- Stretch failures: 5
- Rope-NaN failures: 0
- Final-reward-NaN invalid: 0
- Pair mining uses official-valid full rollouts only: True

## Observation semantics

- Rope observation: artificial post-local partial-state XYZ history
- Claimed deployable: False
- Rope velocity used for selection: False
- Robot observation per arm: EE xyz + EE quaternion + 7 motor-joint qpos
- Post state: fixed shared context
- Local occlusion radius: 0.050000 m
- Visible-history Chamfer threshold: 0.010000 m
- Dual-EE position threshold: 0.010000 m
- Dual-EE quaternion threshold: 0.087266463 rad
- Dual motor-qpos RMS threshold: 0.050000 rad

## Funnel

- total_rollouts: 128
- official_valid_rollouts: 123
- official_invalid_rollouts: 5
- total_valid_same_time_pair_comparisons: 742797
- hidden_winding_index_different: 16012
- nonempty_partial_rope_history: 16012
- robot_ee_position_pass: 13636
- robot_ee_quaternion_pass: 13636
- robot_motor_qpos_pass: 13636
- robot_observation_pass: 13636
- visible_history_chamfer_pass: 5534
- candidate_count: 5534

## Hidden state

- Descriptor: rounded signed task-native winding index `[w1,w2,w3]`
- Candidate records include per-post winding integer residuals for both states and pair max residual.

## Next action

Proceed to PB3 snapshot same-action multi-horizon causal audit. Compare pair divergence against deterministic repeat/uncertainty. Do not reuse PB2-C discovery thresholds as Gate 4.
