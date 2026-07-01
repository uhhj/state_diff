# Stage 1: Hidden Lateral-Jam Hose Insertion

This folder stores generated reports for the stage-1 MuJoCo environment.

## Purpose

The environment is designed for a CCDA pre-experiment:

* Similar visible hose/gripper/socket state at audit time.
* Similar mocap/gripper proprioception.
* Similar scripted action history.
* Different hidden contact condition.
* Different future hose deformation branch.
* Different success outcome.

## Conditions

| Condition          | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| `free_insert`      | Smooth socket, no hidden lateral block.                            |
| `right_hidden_jam` | Hidden lateral friction block inside the right side of the socket. |

## Main scripts

Run smoke test:

```bash
python state_diff/scripts/ccda_hose_stage1_smoke.py
```

Collect debug rollouts and generate a Markdown report:

```bash
python state_diff/scripts/ccda_hose_stage1_collect.py \
  --episodes-per-condition 20 \
  --out data/ccda_hose_stage1/debug_v1.npz \
  --report-dir reports/ccda_hose_stage1/debug_v1
```

Generate videos and comparison plots:

```bash
python state_diff/scripts/ccda_hose_stage1_visualize.py \
  --seed 0 \
  --camera side_top \
  --out-dir reports/ccda_hose_stage1/visual_side_top
```

Generate transparent socket debug videos and trajectory plots:

```bash
python state_diff/scripts/ccda_hose_stage1_visualize_transparent.py \
  --seed 0 \
  --camera side_top \
  --out-dir reports/ccda_hose_stage1/transparent_visual
```

Run test:

```bash
pytest tests/test_ccda_hose_env.py -q
```

## Expected outputs

* `stage1_report.md`
* `summary.json`
* `insertion_depth.png`
* `lateral_offset.png`
* `max_curvature.png`
* `jam_contact_force.png`
* `lateral_contact_force.png`
* rollout videos in MP4 format
* transparent debug videos and plug trajectory CSV

## Notes

This stage does not train State Diffusion. It only validates whether the MuJoCo task can create the hidden-contact future-state branching needed for later CCDA auditing.

The standard visualize script checks the formal occluded condition. The transparent visualize script is only for debugging socket-internal motion: it keeps socket collisions enabled, but makes socket and hidden-jam materials semi-transparent. Transparent videos should not be used as official model inputs or formal CCDA audit observations because they can leak hidden contact geometry. Formal CCDA auditing should still use the non-transparent occluded environment.

`side_top` is the recommended debug camera because it better shows socket entrance, plug motion, front hose keypoints, insertion depth, lateral deflection, and S-bend behavior near the socket.
