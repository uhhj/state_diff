from scripts.experiment3.phase0f_dhr_control_smoke.analyze_control_smoke import (
    classify_smoke,
)


def _config():
    return {
        "criteria": {
            "min_success_progress_m":
                0.03,

            "max_success_peak_force_n":
                20.0,

            "min_jam_left_progress_advantage_m":
                0.015,

            "required_best_candidate": {
                "free":
                    "straight",

                "jam_right":
                    "left_release",
            },

            "selection":
                "lexicographic_success_then_peak_force_then_path_length",
        }
    }


def _row(
        progress,
        force,
        path,
        retained=True):
    return {
        "final_progress_m":
            progress,

        "peak_force_n":
            force,

        "grasp_retained":
            retained,

        "path_length_m":
            path,

        "pre_action_visible_keypoints":
            [[0.0, 0.0, 0.0]],

        "pocket_contact_fraction":
            0.0,

        "pocket_peak_contact_force_n":
            0.0,

        "pocket_contact_beads":
            [],

        "command_phase":
            ["x"],

        "command_ee_target":
            [[0.0, 0.0, 0.0]],

        "command_joint_target":
            [[0.0]],
    }


def _payload(
        matrix):
    return {
        "same_candidate_commands_across_branches":
            True,

        "branch_local_ik":
            False,

        "matrix":
            matrix,
    }


def test_full_matrix_pass_requires_hidden_state_action_switch():
    matrix = {
        "free": {
            "straight":
                _row(
                    0.040,
                    8.0,
                    0.060),

            "left_release":
                _row(
                    0.041,
                    9.0,
                    0.081),

            "right_release":
                _row(
                    0.039,
                    10.0,
                    0.081),
        },

        "jam_right": {
            "straight":
                _row(
                    0.010,
                    25.0,
                    0.060),

            "left_release":
                _row(
                    0.040,
                    11.0,
                    0.081),

            "right_release":
                _row(
                    0.012,
                    24.0,
                    0.081),
        },
    }

    result = classify_smoke(
        _payload(
            matrix
        ),
        _config(),
    )

    assert result[
        "control_structure_pass"
    ] is True

    assert result[
        "best_candidate"
    ][
        "free"
    ] == "straight"

    assert result[
        "best_candidate"
    ][
        "jam_right"
    ] == "left_release"

    assert result[
        "verdict"
    ] == (
        "PHASE0F0_DHR_ACTION_SWITCH_PASS"
    )


def test_rejects_when_free_also_prefers_left_release():
    matrix = {
        "free": {
            "straight":
                _row(
                    0.040,
                    10.0,
                    0.060),

            "left_release":
                _row(
                    0.041,
                    8.0,
                    0.081),

            "right_release":
                _row(
                    0.039,
                    11.0,
                    0.081),
        },

        "jam_right": {
            "straight":
                _row(
                    0.010,
                    25.0,
                    0.060),

            "left_release":
                _row(
                    0.040,
                    11.0,
                    0.081),

            "right_release":
                _row(
                    0.012,
                    24.0,
                    0.081),
        },
    }

    result = classify_smoke(
        _payload(
            matrix
        ),
        _config(),
    )

    assert result[
        "best_candidate"
    ][
        "free"
    ] == "left_release"

    assert result[
        "control_structure_pass"
    ] is False


def test_rejects_when_jam_prefers_right_release():
    matrix = {
        "free": {
            "straight":
                _row(
                    0.040,
                    8.0,
                    0.060),

            "left_release":
                _row(
                    0.041,
                    9.0,
                    0.081),

            "right_release":
                _row(
                    0.039,
                    10.0,
                    0.081),
        },

        "jam_right": {
            "straight":
                _row(
                    0.010,
                    25.0,
                    0.060),

            "left_release":
                _row(
                    0.040,
                    12.0,
                    0.081),

            "right_release":
                _row(
                    0.041,
                    9.0,
                    0.081),
        },
    }

    result = classify_smoke(
        _payload(
            matrix
        ),
        _config(),
    )

    assert result[
        "best_candidate"
    ][
        "jam_right"
    ] == "right_release"

    assert result[
        "control_structure_pass"
    ] is False
