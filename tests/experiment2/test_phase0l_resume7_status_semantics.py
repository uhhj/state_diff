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
from scripts.experiment2.phase0.run_hidden_routing_gate_phase0l import (
    _first_action_failure,
)


def action(
    phase="preload",
    primitive=(
        "pick_precise_latch_probe"
    ),
):
    return {
        "name": (
            "routing_contact_probe"
            if phase == "preload"
            else "routing_main_pull"
        ),
        "phase": phase,
        "primitive": primitive,
    }


def info(
    *,
    primitive_succeeded,
    task_success,
    episode_terminated,
    termination_reason,
):
    return {
        "ccda_execution_status": {
            "action_executed": True,
            "action_completed": True,
            "primitive_succeeded": (
                primitive_succeeded
            ),
            "task_success": task_success,
            "episode_terminated": (
                episode_terminated
            ),
            "termination_reason": (
                termination_reason
            ),
        }
    }


def test_task_success_does_not_stop_fixed_workflow():
    result = _action_execution_result(
        action(),
        info(
            primitive_succeeded=True,
            task_success=True,
            episode_terminated=True,
            termination_reason=(
                "task_success"
            ),
        ),
    )

    assert result[
        "primitive_succeeded"
    ]
    assert result["task_success"]
    assert result[
        "episode_terminated"
    ]
    assert not result[
        "workflow_should_stop"
    ]


def test_primitive_failure_stops_workflow():
    result = _action_execution_result(
        action(),
        info(
            primitive_succeeded=False,
            task_success=False,
            episode_terminated=True,
            termination_reason=(
                "primitive_failed"
            ),
        ),
    )

    assert not result[
        "primitive_succeeded"
    ]
    assert not result["task_success"]
    assert result[
        "workflow_should_stop"
    ]
    assert result[
        "workflow_stop_reason"
    ] == "primitive_failed"


def metadata(
    result,
    events,
):
    return {
        "action_results": [result],
        "motion_events": events,
    }


def test_preload_failure_uses_probe_acquisition():
    result = _action_execution_result(
        action(),
        info(
            primitive_succeeded=False,
            task_success=False,
            episode_terminated=True,
            termination_reason=(
                "primitive_failed"
            ),
        ),
    )

    failure = _first_action_failure(
        metadata(
            result,
            [{
                "stage": (
                    "routing_probe_"
                    "acquisition"
                ),
                "success": False,
                "failure_reason": (
                    "grasp_failed"
                ),
            }],
        ),
        metadata(
            _action_execution_result(
                action(),
                info(
                    primitive_succeeded=True,
                    task_success=False,
                    episode_terminated=False,
                    termination_reason=None,
                ),
            ),
            [],
        ),
    )

    assert failure[
        "phase"
    ] == "preload"
    assert failure[
        "failed_check"
    ] == "preload_action_execution"
    assert failure[
        "failure_reason"
    ] == "grasp_failed"


def test_main_failure_uses_tension_acquisition():
    main_action = action(
        phase="main_pull",
        primitive=(
            "pick_precise_tension_extension"
        ),
    )
    result = _action_execution_result(
        main_action,
        info(
            primitive_succeeded=False,
            task_success=False,
            episode_terminated=True,
            termination_reason=(
                "primitive_failed"
            ),
        ),
    )

    failure = _first_action_failure(
        metadata(
            result,
            [{
                "stage": (
                    "tension_pull_acquisition"
                ),
                "success": False,
                "failure_reason": (
                    "contact_not_detected"
                ),
            }],
        ),
        metadata(
            result,
            [],
        ),
    )

    assert failure[
        "phase"
    ] == "main_pull"
    assert failure[
        "failed_check"
    ] == (
        "main_pull_action_execution"
    )
    assert failure[
        "failure_reason"
    ] == "contact_not_detected"
