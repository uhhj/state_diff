# Stage 1 Visual Debug

Camera: `side_top`

Generated videos:

* `free_insert_side_top.mp4`
* `right_hidden_jam_side_top.mp4`

Generated comparison plots:

* `insertion_depth_compare.png`
* `lateral_offset_compare.png`
* `max_curvature_compare.png`
* `lateral_contact_force_compare.png`

The dashed vertical line in each plot marks the audit time, defined as the end of the approach phase and the start of the shared push phase.

## Frame Diagnostics

The script prints frame-diff and initial/final plug and insertion-depth values for each rollout. If frame-diff is below `1.0`, the rendered video may look static even if the simulator state changes.
