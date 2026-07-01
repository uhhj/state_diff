# Stage 1 Transparent Socket Visual Debug

Camera: `side_top`

This transparent socket environment is only for debug visualization. It should not be used as formal model input and does not change the CCDA data definition.

## Debug Scope

* Socket wall geoms keep collision enabled; only their material alpha changes.
* The hidden jam block is semi-transparent red to show the right-side jam location.
* The transparent visual view can reveal hidden contact and should not be used for official CCDA audit inputs.
* The `side_top` camera is the recommended debug view for socket, plug, hose-front, insertion, and lateral-jam motion.
* If a video still looks static, first inspect plug x and insertion depth over time.

## Generated Files

* `free_insert_side_top_transparent.mp4`
* `right_hidden_jam_side_top_transparent.mp4`
* `hose_keypoint_trajectory_xy.png`
* `hose_keypoint_trajectory_xz.png`
* `plug_trajectory_xyz.csv`

## Frame Diagnostics

* `free_insert`: frame_diff=2.477, initial_depth=0.000000, final_depth=0.110451, initial_plug_pos=[-0.05483974808865435, -0.00010571366103339544, 0.04539124015956515], final_plug_pos=[0.09845092876225858, -6.187573208555425e-10, 0.044995374494138735], final_branch=`success_insert`, final_success=True
* `right_hidden_jam`: frame_diff=2.660, initial_depth=0.000000, final_depth=0.108736, initial_plug_pos=[-0.05448083817381914, 0.0006574642841978553, 0.0451640403810555], final_plug_pos=[0.09673599404002849, -0.0024274186585615627, 0.04498167310170981], final_branch=`lateral_jam`, final_success=False
