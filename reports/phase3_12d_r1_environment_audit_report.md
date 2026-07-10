# Phase3.12d-r1 Physical Environment / Task Semantics Audit

## Verdict

- Verdict: `FAIL`
- Root cause: `phase312d_r1_environment_semantics_or_task_actionability_failed`
- Query-local snapshot allowed: `False`

## Source checks

| Check | Result |
|---|---:|
| `environment_has_physics_hook_dispatch` | `True` |
| `task_has_physics_step_hook` | `True` |
| `task_tracks_physics_steps` | `True` |
| `task_logs_release_physics_step` | `True` |
| `task_still_checks_breakaway_from_reward` | `True` |
| `hidden_geometry_is_collision_only` | `True` |
| `task_uses_world_point_constraint` | `True` |
| `base_cable_is_bead_chain` | `True` |

## Visible-pair probe

| Seed | Bead XY max abs | Bead XY MAE | Fraction diff | Curve diff |
|---:|---:|---:|---:|---:|
| 312000 | 0.04151818 | 0.01444355 | 0.16666667 | 0.06794891 |
| 312500 | 0.05511976 | 0.01491796 | 0.00000000 | 0.15999356 |

## Breakaway physics probe

```json
{
  "force": 15.0,
  "hook_error": "None",
  "max_displacement": 0.04568465496034095,
  "max_steps": 2400,
  "release_loop_step": 15,
  "released_before_reward": true,
  "seed": 312000,
  "task_release_physics_step": 15,
  "threshold": 0.045
}
```

## Breakaway thread-dispatch probe

```json
{
  "force": 15.0,
  "hook_error": "None",
  "release_iteration": 14,
  "released_before_reward": true,
  "seed": 312000,
  "task_release_physics_step": 17
}
```

## Snapshot constraint probe

```json
{
  "active_constraint_id": 26,
  "fallback": {
    "existing": false,
    "new_id": 26,
    "old_id": 25,
    "recreated": true,
    "required": true
  },
  "original_constraint_id": 25,
  "pass": true,
  "restored_by_pybullet": false,
  "seed": 312000
}
```

## Goal action probe

| Seed | Dense gain | Ordered gain | Fraction gain |
|---:|---:|---:|---:|
| 312000 | 0.211906 | 0.141097 | 0.416667 |
| 312500 | 0.558830 | 0.332362 | 0.125000 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `breakaway_is_world_anchor_approximation` | This is a controlled latent constraint, not a geometrically modeled pin contact. |
| `FAIL` | `hidden_condition_changes_initial_visible_geometry_too_much` | [{'seed': 312000, 'bead_xy_max_abs': 0.04151817682833325, 'bead_xy_mae': 0.014443553885193791, 'fraction_diff': 0.16666666666666666, 'curve_diff': 0.06794890565290679}, {'seed': 312500, 'bead_xy_max_abs': 0.0551197642055381, 'bead_xy_mae': 0.014917961860883315, 'fraction_diff': 0.0, 'curve_diff': 0.15999356497453}] |

## Decision

- A FAIL blocks Phase3.12d-r1 candidate execution.
- A WARN records modeling approximations but does not invalidate the oracle audit.
- The beaded cable and world-anchor pin are acceptable only as a controlled qualitative benchmark, not as a high-fidelity silicone-cable model.
