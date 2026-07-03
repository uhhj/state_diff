# Phase3 Code Reading Notes

## Repository State

- Main repository path: `/data/state_diff2`
- Main branch: `Experiment1`
- Main commit before Phase3 implementation: `ba600d61b577fd4873b13575d8cfb1dbe72ce360`
- DeformableRavens submodule branch: `ccda-cable`
- DeformableRavens submodule commit before Phase3 robot-proxy commit: `1677f3f3dbd8913cc585bcc3d07a8795cdc43537`
- Raw code-reading log: `/tmp/phase3_code_reading_raw.log`
- Phase3 robot-proxy submodule commit: `b95180593d3366b754a4d8a4488d723bc9d61be5`

## Core StateDiff Files

The original StateDiff reproduction files were inspected before Phase3 implementation. Phase3 does not modify:

- `train.py`
- `conda_environment.yaml`
- `demo_push_data_collection.py`
- existing files under `state_diff/`

Current core diff check result:

```text
none
```

## Phase1/Phase2 DeformableRavens Task

- Phase1 task file exists: `yes`
- `hidden-contact-cable-line` registered in `ravens.tasks.names`: `yes`
- Active conditions are `free`, `hidden_pin`, and `hidden_high_friction`; `hidden_side_jam` remains removed.

Relevant extras/task-field scan:

```text
49:        self.hidden_condition = os.environ.get("CCDA_HIDDEN_CONDITION", "free")
50:        self.ccda_visible_seed = os.environ.get("CCDA_VISIBLE_SEED", "")
51:        self.ccda_pair_group = os.environ.get("CCDA_PAIR_GROUP", "")
68:        condition = os.environ.get("CCDA_HIDDEN_CONDITION", self.hidden_condition)
69:        self.hidden_condition = condition
70:        self.ccda_visible_seed = os.environ.get("CCDA_VISIBLE_SEED", self.ccda_visible_seed)
71:        self.ccda_pair_group = os.environ.get("CCDA_PAIR_GROUP", self.ccda_pair_group)
73:        if self.hidden_condition not in self.CONDITIONS:
76:                    self.hidden_condition, self.CONDITIONS
84:            "condition": self.hidden_condition,
106:    def _robot_pose_proxy(self):
171:            "hidden_condition": self.hidden_condition,
172:            "ccda_visible_seed": self._safe_int_or_str(self.ccda_visible_seed),
173:            "ccda_pair_group": self.ccda_pair_group,
176:            "robot_pose_proxy": self._robot_pose_proxy(),
178:            "bead_positions": [x["position"] for x in bead_states],
180:            "bead_velocities": [x["linear_velocity"] for x in bead_states],
211:        if self.hidden_condition == "free":
222:        if self.hidden_condition == "hidden_pin":
224:        elif self.hidden_condition == "hidden_high_friction":
227:            raise ValueError(self.hidden_condition)
```

## Existing Extras Before Phase3

The task already logged the following audit metadata in `info['extras']`:

- `bead_positions`
- `bead_velocities`
- `hidden_condition`
- `ccda_visible_seed`
- `ccda_pair_group`
- `hidden_contact_meta` for auditing only

## Robot Pose / Proprioception Proxy

The code reading found that the task did not previously log a robot pose/proprioception proxy. Phase3 adds `robot_pose_proxy` in `ravens/tasks/ccda_hidden_contact_cable.py` only. The proxy is contact-blind and never includes `hidden_condition`, `hidden_contact_meta`, pin ids, or success labels.

Proxy source policy:

- `pybullet_robot_body`: if a robot body id is available from the environment, log joint positions, joint velocities, end-effector position, and end-effector orientation.
- `missing_zero_proxy`: if no robot body id is exposed, log a fixed zero vector with `ee_orientation=[0, 0, 0, 1]`.

Phase3 window preparation records `robot_pose_proxy_source` and reports source counts in `reports/phase3_input_leakage_report.md`.

## Phase3 Input Contract

The Phase3 model inputs are built outside the original StateDiff code:

- `paper_state`: state history containing bead xy, bead velocity xy, and robot pose/proprio proxy.
- `state_action`: the same state history plus past action history.

The following are explicitly excluded from model inputs:

- hidden condition labels
- hidden contact metadata
- pin ids / hidden body ids / hidden constraint ids
- success labels
- future actions

## Working Tree Snapshot During Reading

Main repo status before Phase3 additions:

```text
M .gitignore
 m external/deformable-ravens
?? ccda_phase3/
?? reports/phase3_code_reading_notes.md
?? scripts/phase3_aggregate_folds.py
?? scripts/phase3_check_input_leakage.py
?? scripts/phase3_eval_baselines.py
?? scripts/phase3_generate_large_dataset.sh
?? scripts/phase3_policy_rollout.py
?? scripts/phase3_prepare_windows.py
?? scripts/phase3_run_all.sh
?? scripts/phase3_train_baselines.py
?? scripts/phase3_visualize_failures.py
```

Submodule status during reading / before commit:

```text
M ravens/tasks/ccda_hidden_contact_cable.py
```
