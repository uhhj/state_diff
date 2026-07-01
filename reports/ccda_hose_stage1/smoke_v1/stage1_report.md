# Stage 1 Report: Hidden Lateral-Jam Hose Insertion

## Goal

This report summarizes the MuJoCo stage-1 environment for CCDA auditing. The environment creates two visually similar conditions at audit time:

- `free_insert`: no hidden side contact.
- `right_hidden_jam`: an occluded lateral friction block inside the socket creates hidden contact.

The stage-1 goal is not policy training. It is to verify that the environment can generate visible/proprio/action similarity at audit time, but different hidden contact, different future hose states, and different success outcomes.

## Dataset

Saved debug dataset:

`data/ccda_hose_stage1/smoke_v1.npz`

## Summary

| Condition | N | Success rate | Mean final insertion depth | Mean final lateral offset | Mean final max curvature | Mean max jam force | Mean max lateral force |
|---|---:|---:|---:|---:|---:|---:|---:|
| `free_insert` | 4 | 1.000 | 0.1105 | -0.0000 | 2.39 | 0.000 | 0.000 |
| `right_hidden_jam` | 4 | 0.000 | 0.1088 | -0.0024 | 5.25 | 54.137 | 45.372 |

## Branch counts

```json
{
  "free_insert": {
    "failed_insert": 0,
    "half_insert_wrong_angle": 0,
    "lateral_jam": 0,
    "s_buckle": 0,
    "success_insert": 4
  },
  "right_hidden_jam": {
    "failed_insert": 0,
    "half_insert_wrong_angle": 0,
    "lateral_jam": 4,
    "s_buckle": 0,
    "success_insert": 0
  }
}
```

## Figures

* `insertion_depth.png`
* `lateral_offset.png`
* `max_curvature.png`
* `jam_contact_force.png`
* `lateral_contact_force.png`

## Go / No-Go criteria for Stage 2

Proceed to data collection if:

1. `free_insert` success rate is high.
2. `right_hidden_jam` produces lateral jam, S-buckle, or half-insertion failures.
3. The hidden-jam condition has higher lateral/jam contact force.
4. Future state divergence is visible after the audit time.
5. Audit-time visual state remains similar enough for CCDA pair mining.

If `right_hidden_jam` always completely blocks insertion, reduce `jam_friction` or shrink `jam_block_size`.
If `free_insert` fails often, enlarge `socket_half_width`, reduce wall friction, or increase hose stiffness.
