# Stage 1.5 Calibrated Transparent Socket Visual Debug

## Scope

This script uses the calibrated `HoseEnvConfig` from `reports/ccda_hose_stage1/calibration_v1/best_config.json` and the transparent socket debug environment.
It is only for inspecting socket-internal plug and hose motion. It is not formal model input and does not change the CCDA data definition.

Previous unqualified transparent visual outputs were removed from:

* `reports/ccda_hose_stage1/transparent_debug_close`
* `reports/ccda_hose_stage1/transparent_socket_v2`

## Runtime

* camera: `debug_close`
* transparent_socket: `True`
* show_occluder: `False`

## Calibrated Config Summary

```json
{
  "hose_joint_stiffness": 0.02,
  "max_delta_per_env_step": 0.0008,
  "push_distance": 0.12,
  "push_steps": 140,
  "jam_block_x": 0.015,
  "jam_block_y": 0.0,
  "jam_block_size": [
    0.09,
    0.009,
    0.011
  ],
  "jam_friction": [
    18.0,
    0.3,
    0.08
  ],
  "lateral_offset_threshold": 0.002
}
```

## Generated Files

* `free_insert_debug_close_transparent_calibrated.mp4`
* `right_hidden_jam_debug_close_transparent_calibrated.mp4`
* `plug_trajectory_xy.png`
* `plug_trajectory_xz.png`
* `hose_front_keypoints_xy.png`
* `hose_front_keypoints_xz.png`
* `plug_trajectory_xyz.csv`
* `transparent_calibrated_visual_report.md`

## Diagnostics

| Condition | frame_diff | initial depth | final depth | initial plug | final plug | final branch | final success | max jam force | max lateral force | final lateral offset | final curvature |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `free_insert` | 5.258 | 0.000000 | 0.104820 | `[-0.05483827282684764, -0.00010568708201995117, 0.04541979778018691]` | `[0.0928202003600747, -7.666197926519616e-06, 0.04502690550505849]` | `success_insert` | True | 0.000000 | 0.000000 | -0.000008 | 5.520411 |
| `right_hidden_jam` | 7.787 | 0.000000 | 0.105212 | `[-0.05592166472386386, -0.0012048026995765116, 0.045201715739148385]` | `[0.09321155090295953, -0.0026998967042671774, 0.045011350355125164]` | `half_insert_wrong_angle` | False | 89.912787 | 82.482546 | -0.002700 | 28.514118 |

If a video looks static, inspect `plug_trajectory_xyz.csv` and the frame_diff values above first.
