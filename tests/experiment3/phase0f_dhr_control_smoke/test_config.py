import json
from pathlib import Path


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)


def test_dhr_smoke_config_is_control_only():
    config = json.loads(
        (
            REPO_ROOT
            / "configs/experiment3/phase0f/"
            "dhr_control_smoke.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert config[
        "task_name"
    ] == "ccda-dhr-cable-smoke"

    assert config[
        "dataset_role"
    ] == "benchmark_smoke"

    assert (
        "probe_delta_xyz_m"
        not in config["motion"]
    )

    assert config[
        "execution"
    ]["post_test_steps"] == 1200

    assert config[
        "criteria"
    ]["min_success_progress_m"] == 0.03

    assert config[
        "criteria"
    ]["max_success_peak_force_n"] == 20.0

    assert config[
        "criteria"
    ][
        "min_jam_left_progress_advantage_m"
    ] == 0.015

    assert config[
        "criteria"
    ][
        "required_best_candidate"
    ] == {
        "free":
            "straight",

        "jam_right":
            "left_release",
    }

    assert config[
        "criteria"
    ][
        "selection"
    ] == (
        "lexicographic_success_then_peak_force_then_path_length"
    )
