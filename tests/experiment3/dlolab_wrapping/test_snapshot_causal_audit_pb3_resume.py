import json
from pathlib import Path

import pytest

from scripts.experiment3.dlolab_wrapping.paths import REPO_ROOT
from scripts.experiment3.dlolab_wrapping.snapshot_causal_audit_pb3 import (
    PB3Blocked,
    enforce_live_pair_barrier,
    enforce_targeted_alignment_barrier,
)


def _alignments(valid=True):
    return [
        {
            "rollout_id": rollout_id,
            "time_index": 13 if rollout_id % 2 == 0 else 20,
            "valid": bool(valid or rollout_id != 19),
        }
        for rollout_id in range(20)
    ]


def _pairs(valid=True):
    return [
        {
            "pair_id": f"pb3_pair_{index:02d}",
            "valid": bool(valid or index != 9),
        }
        for index in range(10)
    ]


def test_targeted_alignment_barrier_requires_all_20_before_future():
    enforce_targeted_alignment_barrier(_alignments())

    with pytest.raises(PB3Blocked) as caught:
        enforce_targeted_alignment_barrier(_alignments(valid=False))

    assert caught.value.verdict == "PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED"
    assert caught.value.details["failure_component"] == (
        "targeted_replay_alignment"
    )
    assert caught.value.details["future_suffix_executed"] is False


def test_live_pair_barrier_requires_10_of_10_without_replacement():
    enforce_live_pair_barrier(_pairs())

    with pytest.raises(PB3Blocked) as caught:
        enforce_live_pair_barrier(_pairs(valid=False))

    assert caught.value.verdict == "PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED"
    assert caught.value.details["failure_component"] == (
        "live_pair_revalidation"
    )
    assert caught.value.details["actual_pair_count"] == 10
    assert caught.value.details["future_suffix_executed"] is False


def test_resume_config_keeps_r3_tolerance_out_of_gate4():
    config = json.loads(
        (
            REPO_ROOT
            / "configs/experiment3/published_benchmark/"
            "dlolab_wrapping_pb3_resume.json"
        ).read_text()
    )
    rule = json.loads(
        (
            REPO_ROOT
            / "configs/experiment3/published_benchmark/"
            "dlolab_wrapping_pb3r3_alignment_rule.json"
        ).read_text()
    )

    rope_threshold = rule["targeted_replay_engineering_alignment"]["rope"][
        "threshold_m"
    ]
    assert rope_threshold == pytest.approx(61.945756736e-6, rel=1e-11)
    assert rule["snapshot_restore_alignment"]["threshold_m"] == 50e-6
    assert config["gate4"]["absolute_effect_min_m"] == 1e-3
    assert config["gate4"]["repeat_floor_multiplier"] == 5.0
    assert rope_threshold != config["gate4"]["absolute_effect_min_m"]


def test_resume_rule_forbids_pair_drop_and_replacement():
    rule_path = (
        Path(REPO_ROOT)
        / "configs/experiment3/published_benchmark/"
        "dlolab_wrapping_pb3r3_alignment_rule.json"
    )
    rule = json.loads(rule_path.read_text())
    pair_rule = rule["live_pair_revalidation"]

    assert pair_rule["all_10_formal_pairs_required"] is True
    assert pair_rule["allow_pair_drop"] is False
    assert pair_rule["allow_pair_replacement"] is False
