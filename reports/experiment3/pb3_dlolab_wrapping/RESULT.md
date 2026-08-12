# PB3 DLO-Lab Wrapping Same-Action Causal Bifurcation Audit

Verdict: `PB3_TARGETED_REPLAY_ALIGNMENT_FAILED`

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

## Blocked details

```json
{
  "failed_alignment": {
    "rollout_id": 46,
    "time_index": 13,
    "valid": false,
    "rope_max_abs_m": 5.182623863220215e-05,
    "ee_max_abs_m": 3.8743019104003906e-07,
    "motor_qpos_max_abs_rad": 3.5762786865234375e-07,
    "live_winding_index": [
      0,
      0,
      0
    ],
    "frozen_winding_index": [
      0,
      0,
      0
    ]
  }
}
```

## Next action

Fix exact PB2-C targeted replay alignment only. Do not interpret future bifurcation and do not loosen the alignment tolerance automatically.

## Boundary

PB3 tests same-action future dynamics only. It does not establish control relevance, deployable sensing, or model performance.
