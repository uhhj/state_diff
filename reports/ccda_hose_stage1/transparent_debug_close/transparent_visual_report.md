# Stage 1 Transparent Socket Visual Debug

Camera: `debug_close`

transparent_socket: `True`

show_occluder: `False`

This transparent socket environment is only for debug visualization. It should not be used as formal model input and does not change the CCDA data definition.

## Debug Scope

* Socket collision wall geoms keep collision enabled and are rendered nearly invisible.
* Separate visual-only socket walls use `contype="0" conaffinity="0"` and transparent blue rgba.
* The hidden jam block is semi-transparent red to show the right-side jam location.
* The transparent visual view can reveal hidden contact and should not be used for official CCDA audit inputs.
* The `debug_close` camera is the recommended close-up debug view for socket-internal plug and hose motion.
* The `side_top` camera remains useful for wider context.
* If a video still looks static, first inspect plug x and insertion depth over time.

## Generated Files

* `free_insert_debug_close_transparent.mp4`
* `right_hidden_jam_debug_close_transparent.mp4`
* `plug_trajectory_xy.png`
* `plug_trajectory_xz.png`
* `hose_front_keypoints_xy.png`
* `hose_front_keypoints_xz.png`
* `plug_trajectory_xyz.csv`

## Frame Diagnostics

* `free_insert`: frame_diff=5.081, initial_depth=0.000000, final_depth=0.110451, initial_plug_pos=[-0.05483974808865435, -0.00010571366103339544, 0.04539124015956515], final_plug_pos=[0.09845092876225858, -6.187573208555425e-10, 0.044995374494138735], final_branch=`success_insert`, final_success=True, max_lateral_contact_force=0.000000, max_jam_contact_force=0.000000
* `right_hidden_jam`: frame_diff=5.507, initial_depth=0.000000, final_depth=0.108736, initial_plug_pos=[-0.05448083817381914, 0.0006574642841978553, 0.0451640403810555], final_plug_pos=[0.09673599404002849, -0.0024274186585615627, 0.04498167310170981], final_branch=`lateral_jam`, final_success=False, max_lateral_contact_force=45.347894, max_jam_contact_force=54.072669
