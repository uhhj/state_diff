from scripts.experiment3.phase0_ohj_cable.analyze_control_relevance import (
    classify_control)
from scripts.experiment3.phase0_ohj_cable.common import load_config


CONFIG = load_config("configs/experiment3/phase0/ohj_cable_phase0d.json")


def row(progress, force=2., retained=True):
    return {"final_progress_m": progress, "peak_force_n": force,
            "grasp_retained": retained, "path_length_m": .06,
            "command_phase": ["test"],
            "command_ee_target": [[.1, 0., 0.]],
            "command_joint_target": [[0.] * 6]}


def test_free_straight_and_jam_left_pattern_is_control_relevant():
    matrix = {
        "free": {"straight": row(.04), "left_release": row(.035),
                 "right_release": row(.02)},
        "jam_right": {"straight": row(.01), "left_release": row(.04),
                      "right_release": row(.005)}}
    metrics = classify_control(matrix, CONFIG)
    assert metrics["verdict"] == "PHASE0D_CONTROL_RELEVANCE_COMPLETE"
    assert metrics["same_candidate_commands_across_branches"]


def test_same_successful_straight_action_is_not_control_relevant():
    matrix = {
        "free": {"straight": row(.04), "left_release": row(.03),
                 "right_release": row(.02)},
        "jam_right": {"straight": row(.04), "left_release": row(.04),
                      "right_release": row(.02)}}
    assert classify_control(matrix, CONFIG)["verdict"] == (
        "PHASE0D_CONTROL_NOT_RELEVANT")
