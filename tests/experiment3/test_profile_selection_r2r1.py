import pytest

from scripts.experiment3.phase0_soft_blockpush_r2r1.common import (
    SOFTER_PROFILE, START_PROFILE, STIFFER_PROFILE, choose_next_profile)


@pytest.mark.parametrize("verdict,expected", [
    ("PHASE0B_R2_AXIAL_TOO_STIFF", SOFTER_PROFILE),
    ("PHASE0B_R2_AXIAL_TOO_SOFT", STIFFER_PROFILE),
    ("PHASE0B_R2_SHEAR_TOO_STIFF", SOFTER_PROFILE),
    ("PHASE0B_R2_SHEAR_TOO_SOFT", STIFFER_PROFILE)])
def test_start_profile_moves_to_one_matching_neighbor(verdict, expected):
    decision = choose_next_profile(START_PROFILE, verdict, 1)
    assert not decision.stop and decision.next_profile_name == expected


@pytest.mark.parametrize("verdict", [
    "PHASE0B_R2_AXIAL_COMPLETE", "PHASE0B_R2_AXIAL_UNSTABLE",
    "PHASE0B_R2_AXIAL_ENGINEERING_BLOCKED"])
def test_complete_or_engineering_failure_does_not_switch(verdict):
    decision = choose_next_profile(START_PROFILE, verdict, 1)
    assert decision.stop and decision.next_profile_name is None


def test_second_profile_and_two_profile_limit_never_allow_a_third():
    assert choose_next_profile(SOFTER_PROFILE, "PHASE0B_R2_AXIAL_TOO_SOFT", 1).stop
    assert choose_next_profile(START_PROFILE, "PHASE0B_R2_AXIAL_TOO_STIFF", 2).stop
