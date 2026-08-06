import numpy as np

from scripts.experiment3.phase0_soft_blockpush.analyze_pair import classify_verdict
from state_diff.env.block_pushing.soft_block_metrics import (
    edge_strain, first_sustained_onset, rigid_aligned_rmse)


def test_kabsch_translation_rotation_reflection_and_deformation():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    theta = np.pi / 3
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moved = points @ rotation.T + np.array([2., -3., .5])
    assert rigid_aligned_rmse(points, moved) < 1e-12
    reflected = points.copy(); reflected[:, 0] *= -1
    assert rigid_aligned_rmse(points, reflected) > 0.1
    deformed = points.copy(); deformed[-1, 2] += .3
    assert rigid_aligned_rmse(points, deformed) > 0.01


def test_edge_strain_and_sustained_onset():
    points = np.array([[0., 0., 0.], [1.1, 0., 0.]])
    assert np.allclose(edge_strain(points, np.array([[0, 1]]), np.array([1.])), .1)
    values = np.array([0., .1, 5.1, 5.2, 5.3, 0.])
    eligible = np.array([False, False, True, True, True, True])
    assert first_sustained_onset(values, 5., eligible, 3) == 2


def test_translation_only_and_complete_verdicts():
    translation = {"visible": True, "full": True, "rigid": False,
                   "progress": True, "amplification": True, "edge_ratios": True}
    verdict, cause = classify_verdict(True, True, True, True, translation)
    assert verdict == "PHASE0B_SINGLE_PAIR_SCIENTIFIC_FAIL"
    assert cause == "translation_only_no_deformation_branch"
    complete = dict(translation, rigid=True)
    assert classify_verdict(True, True, True, True, complete)[0] == "PHASE0B_SINGLE_PAIR_COMPLETE"
