"""Classify the full DHR 2x3 action-switch smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from scripts.experiment3.phase0_ohj_cable.analyze_control_relevance import (  # noqa: E402
    classify_control,
)
from scripts.experiment3.phase0_ohj_cable.common import (  # noqa: E402
    write_json,
)
from scripts.experiment3.phase0f_dhr_control_smoke.common import (  # noqa: E402
    load_config,
)


def _git(
        *args):
    return subprocess.check_output(
        [
            "git",
            *args,
        ],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _legacy_config(
        config):
    return {
        "analysis": {
            "min_extraction_progress_m":
                config[
                    "criteria"
                ][
                    "min_success_progress_m"
                ],

            "max_bruteforce_wrench_force_n":
                config[
                    "criteria"
                ][
                    "max_success_peak_force_n"
                ],
        }
    }


def classify_smoke(
        payload,
        config):
    matrix = payload[
        "matrix"
    ]

    legacy = classify_control(
        matrix,
        _legacy_config(
            config
        ),
    )

    required = config[
        "criteria"
    ][
        "required_best_candidate"
    ]

    best = legacy[
        "best_candidate"
    ]

    rows = legacy[
        "matrix"
    ]

    jam_advantage = float(
        legacy[
            "jam_left_progress_advantage_m"
        ]
    )

    free_straight = rows[
        "free"
    ][
        "straight"
    ]

    jam_left = rows[
        "jam_right"
    ][
        "left_release"
    ]

    action_switch = bool(
        best[
            "free"
        ]
        == required[
            "free"
        ]
        and best[
            "jam_right"
        ]
        == required[
            "jam_right"
        ]
    )

    control_pass = bool(
        payload[
            "same_candidate_commands_across_branches"
        ]
        and legacy[
            "control_relevant"
        ]
        and free_straight[
            "success"
        ]
        and jam_left[
            "success"
        ]
        and action_switch
        and jam_advantage
        >= config[
            "criteria"
        ][
            "min_jam_left_progress_advantage_m"
        ]
    )

    free_visible = np.asarray(
        matrix[
            "free"
        ][
            "straight"
        ][
            "pre_action_visible_keypoints"
        ],
        dtype=np.float64,
    )

    jam_visible = np.asarray(
        matrix[
            "jam_right"
        ][
            "straight"
        ][
            "pre_action_visible_keypoints"
        ],
        dtype=np.float64,
    )

    visible_rmse = float(
        np.sqrt(
            np.mean(
                (
                    free_visible
                    - jam_visible
                )
                ** 2
            )
        )
    )

    verdict = (
        "PHASE0F0_DHR_ACTION_SWITCH_PASS"
        if control_pass
        else
        "PHASE0F0_DHR_ACTION_SWITCH_FAIL"
    )

    return {
        "verdict":
            verdict,

        "control_structure_pass":
            control_pass,

        "same_candidate_commands_across_branches":
            bool(
                payload[
                    "same_candidate_commands_across_branches"
                ]
            ),

        "branch_local_ik":
            bool(
                payload[
                    "branch_local_ik"
                ]
            ),

        "best_candidate":
            dict(
                best
            ),

        "required_best_candidate":
            dict(
                required
            ),

        "best_action_switch":
            action_switch,

        "legacy_control_relevant":
            bool(
                legacy[
                    "control_relevant"
                ]
            ),

        "jam_left_progress_advantage_m":
            jam_advantage,

        "pre_action_visible_rmse_m":
            visible_rmse,

        "matrix":
            rows,

        "raw_matrix":
            matrix,

        "next_task":
            (
                "Freeze DHR mechanics and implement "
                "Phase 0F1 formal paired CCDA admission."
                if control_pass
                else
                "Reject this DHR benchmark candidate. "
                "Do not tune pocket, route, actions, "
                "or thresholds; review the benchmark "
                "concept at task level."
            ),
    }


def _report_row(
        result,
        condition,
        candidate):
    row = dict(
        result["matrix"][condition][candidate])
    raw = result[
        "raw_matrix"
    ][condition][candidate]
    row.update({
        "pocket_contact_fraction":
            raw["pocket_contact_fraction"],
        "pocket_peak_contact_force_n":
            raw["pocket_peak_contact_force_n"],
        "pocket_contact_beads":
            raw["pocket_contact_beads"],
    })
    return row


def analyze(
        config_path):
    config = load_config(
        config_path
    )

    output_root = Path(
        config["output_root"]
    )

    payload = json.loads(
        (
            output_root
            / "CONTROL_SMOKE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    result = classify_smoke(
        payload,
        config,
    )

    write_json(
        output_root
        / "CONTROL_SMOKE_METRICS.json",
        result,
    )

    committed = (
        REPO_ROOT
        / config[
            "committed_report_dir"
        ]
    )

    committed.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_text = f"""Verdict: {result["verdict"]}

Scientific status:
- Formal Phase 0: NOT RUN
- Model training: NOT RUN
- Control-first benchmark admission: {"PASS" if result["control_structure_pass"] else "FAIL"}

Best action:
- FREE: {result["best_candidate"]["free"]}
- JAM-R: {result["best_candidate"]["jam_right"]}
- Required FREE: {result["required_best_candidate"]["free"]}
- Required JAM-R: {result["required_best_candidate"]["jam_right"]}
- Best-action switch: {result["best_action_switch"]}

Classifier invariants:
- Legacy control relevance: {result["legacy_control_relevant"]}
- Same candidate commands: {result["same_candidate_commands_across_branches"]}
- Branch-local IK: {result["branch_local_ik"]}

FREE:
- straight: {_report_row(result, "free", "straight")}
- left_release: {_report_row(result, "free", "left_release")}
- right_release: {_report_row(result, "free", "right_release")}

JAM-R:
- straight: {_report_row(result, "jam_right", "straight")}
- left_release: {_report_row(result, "jam_right", "left_release")}
- right_release: {_report_row(result, "jam_right", "right_release")}

Key comparison:
- JAM left progress advantage: {result["jam_left_progress_advantage_m"]} m
- Pre-action visible RMSE: {result["pre_action_visible_rmse_m"]} m

Next task:
{result["next_task"]}
"""

    (
        committed
        / "RESULT.md"
    ).write_text(
        result_text,
        encoding="utf-8",
    )

    evidence = {
        "phase_name":
            config[
                "phase_name"
            ],

        "verdict":
            result[
                "verdict"
            ],

        "repository": {
            "starting_main_sha":
                config[
                    "provenance"
                ][
                    "starting_main_sha"
                ],

            "ending_main_sha":
                _git(
                    "rev-parse",
                    "HEAD",
                ),

            "ending_submodule_sha":
                _git(
                    "rev-parse",
                    "HEAD:external/deformable-ravens",
                ),
        },

        "scientific":
            result,

        "training": {
            "B0": False,
            "B1": False,
            "CFPM": False,
            "IDM": False,
        },
    }

    write_json(
        committed
        / "EVIDENCE.json",
        evidence,
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
    )
    args = parser.parse_args()

    result = analyze(
        args.config
    )

    print(
        "verdict={}"
        .format(
            result[
                "verdict"
            ]
        )
    )


if __name__ == "__main__":
    main()
