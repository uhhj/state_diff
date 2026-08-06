from scripts.experiment3.phase0_soft_blockpush_r1.analyze_pair import classify_pair


def _gates(**changes):
    gates = {"visible": True, "full": True, "rigid": True,
             "sustained_rigid": True, "deformation_fraction": True,
             "progress": True, "amplification": True, "edge_ratios": True,
             "force_cap": True, "probe_mechanism": True, "com": True}
    gates.update(changes)
    return gates


def test_probe_requires_local_anchor_and_policy_lead():
    assert classify_pair(True, True, True, False, True, False)[0] == \
        "PHASE0B_R1_PROBE_LOCAL_ANCHOR_FAIL"
    assert classify_pair(True, True, True, True, False, False)[0] == \
        "PHASE0B_R1_PROBE_POLICY_LEAD_FAIL"


def test_full_translation_progress_and_complete_classification():
    verdict, cause = classify_pair(
        True, True, True, True, True, True, _gates(rigid=False))
    assert verdict == "PHASE0B_R1_SINGLE_PAIR_SCIENTIFIC_FAIL"
    assert cause == "translation_only_under_compliant_material"
    verdict, cause = classify_pair(
        True, True, True, True, True, True, _gates(progress=False))
    assert cause == "deformation_branch_without_task_progress"
    assert classify_pair(True, True, True, True, True, True, _gates())[0] == \
        "PHASE0B_R1_SINGLE_PAIR_COMPLETE"
