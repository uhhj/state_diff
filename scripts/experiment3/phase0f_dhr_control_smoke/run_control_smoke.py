"""Run the full DHR 2x3 action-switch smoke."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

SIM_ROOT = (
    REPO_ROOT
    / "external"
    / "deformable-ravens"
)

for root in (
        REPO_ROOT,
        SIM_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(
            0,
            str(root))

import pybullet as p  # noqa: E402
from ravens import Environment, tasks  # noqa: E402
from ravens.tasks.ccda_dhr_geometry import (  # noqa: E402
    DHRGeometryConfig,
)

from scripts.experiment3.phase0_ohj_cable.commands import (  # noqa: E402
    build_candidate_scripts,
    command_arrays,
    execute_fixed_phase,
)
from scripts.experiment3.phase0_ohj_cable.common import (  # noqa: E402
    write_json,
)
from scripts.experiment3.phase0_ohj_cable.run_pair import (  # noqa: E402
    acquire_active_endpoint,
    grasp_retained,
)
from scripts.experiment3.phase0f_dhr_control_smoke.common import (  # noqa: E402
    load_config,
)


def geometry_config(config):
    fields = dict(
        config["geometry"])
    fields.pop(
        "visible_keypoint_indices")
    return DHRGeometryConfig(
        **fields)


def _prepare_branch(
        env,
        config,
        geometry,
        condition):
    task = tasks.names[
        config["task_name"]]()

    task.configure_control_smoke(
        condition=condition,
        run_id=config["run_id"],
        seed=config["seed"],
        settle_seconds=config[
            "execution"
        ]["initial_settle_seconds"],
        geometry_config=geometry)

    env.reset(task)

    active_id = acquire_active_endpoint(
        env,
        task,
        config["motion"])

    env.step_physics(
        config[
            "execution"
        ]["pre_action_settle_steps"])

    reference = (
        task._bead_positions()[:4]
        .mean(axis=0))

    task.reset_branch_runtime(
        condition)

    task.reference_passive_xyz = (
        reference.copy())

    ee_state = p.getLinkState(
        env.ur5,
        env.ee_tip_link,
        computeForwardKinematics=True)

    start = np.asarray(
        ee_state[0],
        dtype=np.float64)

    orientation = np.asarray(
        ee_state[1],
        dtype=np.float64)

    previous = np.asarray([
        p.getJointState(
            env.ur5,
            int(joint))[0]
        for joint
        in env.joints
    ], dtype=np.float64)

    return {
        "task":
            task,

        "active_id":
            active_id,

        "start":
            start,

        "orientation":
            orientation,

        "previous":
            previous,

        "visible":
            task.visible_keypoints()
            .copy(),
    }


def _rollout(
        env,
        config,
        geometry,
        planned,
        condition,
        candidate):
    prepared = _prepare_branch(
        env,
        config,
        geometry,
        condition,
    )

    task = prepared["task"]
    active_id = prepared["active_id"]

    phases = planned[candidate]

    for phase in phases:
        execute_fixed_phase(
            env,
            task,
            phase,
            config["motion"][
                "position_gain"
            ],
        )

    task.set_ccda_phase(
        "post_test"
    )

    env.step_physics(
        config["execution"][
            "post_test_steps"
        ]
    )

    trace = task.ccda_trace()

    wrench = np.asarray(
        [
            row["formal_wrench"]
            for row in trace
        ],
        dtype=np.float64,
    )

    contact_force = np.asarray(
        [
            row[
                "oracle_latch_contact_force"
            ]
            for row in trace
        ],
        dtype=np.float64,
    )

    masks = np.asarray(
        [
            row[
                "oracle_latch_contact_bead_mask"
            ]
            for row in trace
        ],
        dtype=np.int8,
    )

    contacted = np.flatnonzero(
        np.any(
            masks > 0,
            axis=0,
        )
    )

    commands = command_arrays(
        phases
    )

    targets = np.asarray(
        commands["ee_target"],
        dtype=np.float64,
    )

    path_length = float(
        np.sum(
            np.linalg.norm(
                np.diff(
                    np.vstack(
                        [
                            prepared["start"],
                            targets,
                        ]
                    ),
                    axis=0,
                ),
                axis=1,
            )
        )
    )

    return {
        "condition":
            condition,

        "candidate":
            candidate,

        "pre_action_visible_keypoints":
            prepared[
                "visible"
            ].astype(
                float
            ).tolist(),

        "final_progress_m":
            float(
                trace[-1][
                    "extraction_progress_m"
                ]
            ),

        "peak_force_n":
            float(
                np.max(
                    np.linalg.norm(
                        wrench[:, :3],
                        axis=1,
                    )
                )
            ),

        "grasp_retained":
            bool(
                grasp_retained(
                    env,
                    active_id,
                )
            ),

        "path_length_m":
            path_length,

        "pocket_contact_fraction":
            float(
                np.mean(
                    contact_force > 0.0
                )
            ),

        "pocket_peak_contact_force_n":
            float(
                np.max(
                    contact_force
                )
            ),

        "pocket_contact_beads":
            contacted.astype(
                int
            ).tolist(),

        "command_phase":
            commands[
                "phase"
            ].tolist(),

        "command_ee_target":
            commands[
                "ee_target"
            ].astype(
                float
            ).tolist(),

        "command_joint_target":
            commands[
                "joint_target"
            ].astype(
                float
            ).tolist(),
    }


def run(
        config_path):
    config = load_config(
        config_path
    )

    geometry = geometry_config(
        config
    )

    env = Environment(
        disp=False,
        hz=config["execution"]["hz"],
        deterministic=True,
        control_substeps=config[
            "execution"
        ][
            "control_substeps"
        ],
    )

    try:
        planner = _prepare_branch(
            env,
            config,
            geometry,
            "free",
        )

        scripts = build_candidate_scripts(
            env,
            planner["start"],
            planner["orientation"],
            config["motion"],
            planner["previous"],
        )

        candidates = (
            "straight",
            "left_release",
            "right_release",
        )

        planned = {
            name:
                scripts[name]
            for name in candidates
        }

        matrix = {
            "free": {},
            "jam_right": {},
        }

        for condition in (
                "free",
                "jam_right"):
            for candidate in candidates:
                matrix[
                    condition
                ][
                    candidate
                ] = _rollout(
                    env,
                    config,
                    geometry,
                    planned,
                    condition,
                    candidate,
                )

        same_candidate_commands = all(
            matrix["free"][candidate][key]
            == matrix["jam_right"][
                candidate
            ][
                key
            ]
            for candidate in candidates
            for key in (
                "command_phase",
                "command_ee_target",
                "command_joint_target",
            )
        )

        result = {
            "diagnostic_only":
                True,

            "formal_phase0":
                False,

            "branch_local_ik":
                False,

            "same_candidate_commands_across_branches":
                bool(
                    same_candidate_commands
                ),

            "matrix":
                matrix,
        }

        output_root = Path(
            config["output_root"]
        )

        write_json(
            output_root
            / "CONTROL_SMOKE.json",
            result,
        )

        return result

    finally:
        env.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    args = parser.parse_args()

    result = run(
        args.config
    )

    print(
        "same_candidate_commands={}"
        .format(
            result[
                "same_candidate_commands_across_branches"
            ]
        )
    )


if __name__ == "__main__":
    main()
