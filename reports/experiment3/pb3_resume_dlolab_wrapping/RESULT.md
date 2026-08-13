# PB3 DLO-Lab Wrapping Same-Action Causal Bifurcation Audit

Verdict: `PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED`

## Formal design

- Shortlist: 10 PB2-C pairs / 20 unique rollouts
- Shortlist selected without future information: True
- Snapshot repeats per branch state: 3
- Horizons: `[1, 2, 5, 10, 20, 40]`
- Primary metric: ordered rope displacement-field RMSE `(X(t+h)-X(t))_A vs (X(t+h)-X(t))_B`
- Absolute effect minimum: 1.000 mm
- Repeat-floor multiplier: 5.000x

## Replay / snapshot

- Targeted replay alignment valid: False
- Live PB2-C pair revalidation valid: False
- Pre-future barrier passed: False
- Future suffix executed: False

## Blocked details

```json
{
  "failure_component": "targeted_replay_alignment",
  "expected_branch_state_count": 20,
  "actual_branch_state_count": 20,
  "unique_branch_state_count": 20,
  "failed_alignments": [
    {
      "rollout_id": 39,
      "time_index": 20,
      "valid": false,
      "rope_coordinate_rmse_m": 0.00031341783035073774,
      "rope_max_abs_coordinate_m_diagnostic": 0.0015099570155143738,
      "ee_max_abs_m": 1.8775463104248047e-06,
      "motor_qpos_max_abs_rad": 2.1457672119140625e-06,
      "live_winding_index": [
        1,
        1,
        0
      ],
      "frozen_winding_index": [
        1,
        1,
        0
      ]
    },
    {
      "rollout_id": 55,
      "time_index": 20,
      "valid": false,
      "rope_coordinate_rmse_m": 0.0008244199725039843,
      "rope_max_abs_coordinate_m_diagnostic": 0.004064053297042847,
      "ee_max_abs_m": 6.020069122314453e-06,
      "motor_qpos_max_abs_rad": 8.225440979003906e-06,
      "live_winding_index": [
        0,
        1,
        0
      ],
      "frozen_winding_index": [
        1,
        1,
        0
      ]
    },
    {
      "rollout_id": 97,
      "time_index": 20,
      "valid": false,
      "rope_coordinate_rmse_m": 0.0001576423955645504,
      "rope_max_abs_coordinate_m_diagnostic": 0.0008927993476390839,
      "ee_max_abs_m": 1.296401023864746e-06,
      "motor_qpos_max_abs_rad": 4.172325134277344e-06,
      "live_winding_index": [
        0,
        1,
        0
      ],
      "frozen_winding_index": [
        0,
        1,
        0
      ]
    }
  ],
  "future_suffix_executed": false
}
```

## Next action

Stop before all snapshot/future work. Preserve the frozen R3 rule and formal 10-pair shortlist; do not drop or replace pairs.

## Boundary

PB3 tests same-action future dynamics only. It does not establish control relevance, deployable sensing, or model performance.
