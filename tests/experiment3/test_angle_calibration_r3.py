from scripts.experiment3.phase0_soft_blockpush_r3.common import (
    SOFTER_PROFILE, STIFFER_PROFILE, choose_next_profile)


def test_only_shear_softness_selects_one_adjacent_profile():
    assert choose_next_profile("PHASE0B_R2_SHEAR_TOO_SOFT") == STIFFER_PROFILE
    assert choose_next_profile("PHASE0B_R2_SHEAR_TOO_STIFF") == SOFTER_PROFILE
    assert choose_next_profile("PHASE0B_R2_SHEAR_UNSTABLE") is None
    assert choose_next_profile("PHASE0B_R2_AXIAL_TOO_SOFT") is None
