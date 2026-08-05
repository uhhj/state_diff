import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_ROOT = (
    ROOT
    / "external"
    / "deformable-ravens"
)
for path in (
    ROOT,
    SUBMODULE_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )

from scripts.experiment2.phase0.exact_counterfactual import (
    _action_execution_result,
)
from scripts.experiment2.phase0.run_hidden_friction_pairs import (
    _environment_action,
)
from scripts.experiment2.phase0.run_hidden_routing_gate_phase0l import (
    _first_action_failure,
)


def action():
    return {
        "name": "routing_main_pull",
        "phase": "main_pull",
        "primitive": (
            "pick_precise_tension_extension"
        ),
    }


def test_failed_environment_step_is_preserved():
    result = _action_execution_result(
        action(),
        True,
        {
            "extras": {
                "task.done": False,
            }
        },
    )
    assert not result[
        "primitive_succeeded"
    ]
    assert result["returned_done"]
    assert not result["task_done"]


def test_successful_environment_step_is_preserved():
    result = _action_execution_result(
        action(),
        False,
        {
            "extras": {
                "task.done": False,
            }
        },
    )
    assert result[
        "primitive_succeeded"
    ]
    assert not result["returned_done"]


def test_runner_returns_acquisition_reason():
    free = {
        "action_results": [
            {
                **action(),
                "action_name": (
                    "routing_main_pull"
                ),
                "primitive_succeeded": (
                    False
                ),
                "returned_done": True,
                "task_done": False,
                "exit_gracefully": False,
            }
        ],
        "motion_events": [
            {
                "stage": (
                    "tension_pull_"
                    "acquisition"
                ),
                "success": False,
                "failure_reason": (
                    "grasp_failed"
                ),
            }
        ],
    }
    hidden = {
        "action_results": [],
        "motion_events": [],
    }

    failure = _first_action_failure(
        free,
        hidden,
    )
    assert failure["condition"] == "free"
    assert failure["acquisition"][
        "failure_reason"
    ] == "grasp_failed"


def test_environment_action_passes_acquisition_mode():
    params = _environment_action({
        "name": "routing_main_pull",
        "phase": "main_pull",
        "primitive": (
            "pick_precise_tension_extension"
        ),
        "pose0": {
            "position": [0.4, -0.3, 0.001],
            "quaternion": [0, 0, 0, 1],
        },
        "pose_stage1": {
            "position": [0.4, -0.22, 0.001],
            "quaternion": [0, 0, 0, 1],
        },
        "pose1": {
            "position": [0.4, -0.18, 0.001],
            "quaternion": [0, 0, 0, 1],
        },
        "lift_height": 0.004,
        "approach_height": 0.020,
        "retreat_z": 0.300,
        "joint_tolerance": 0.0001,
        "cartesian_tolerance": 0.0002,
        "min_achieved_fraction": 0.8,
        "acquisition_motion_mode": (
            "precise_endpoint_recovery"
        ),
    })
    assert params["params"][
        "acquisition_motion_mode"
    ] == "precise_endpoint_recovery"
