import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client
import pytest

from state_diff.env.block_pushing.kelvin_voigt_soft_block import (
    KelvinVoigtMaterial, KelvinVoigtSoftBlock, kelvin_voigt_edge_force)
from state_diff.env.block_pushing.soft_block_lattice import (
    SoftBlockConfig, build_edge_metadata, build_edge_pairs)


def _force(a, va, b, vb, rest=1., k=2., c=.5, cap=10.):
    return kelvin_voigt_edge_force(
        np.asarray(a, float), np.asarray(va, float), np.asarray(b, float),
        np.asarray(vb, float), rest, k, c, cap)


def test_edge_force_sign_damping_cap_energy_and_collapse():
    force, capped, energy = _force([0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0])
    assert np.array_equal(force, np.zeros(3)) and not capped and energy == 0
    stretched = _force([0, 0, 0], [0, 0, 0], [1.2, 0, 0], [0, 0, 0])[0]
    compressed = _force([0, 0, 0], [0, 0, 0], [.8, 0, 0], [0, 0, 0])[0]
    assert stretched[0] > 0 and compressed[0] < 0
    separation = _force([0, 0, 0], [-.1, 0, 0], [1, 0, 0], [.1, 0, 0])[0]
    approach = _force([0, 0, 0], [.1, 0, 0], [1, 0, 0], [-.1, 0, 0])[0]
    assert separation[0] > 0 and approach[0] < 0
    force, capped, energy = _force(
        [0, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0], cap=.1)
    assert capped and np.isclose(np.linalg.norm(force), .1) and energy >= 0
    with pytest.raises(FloatingPointError):
        _force([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])


def test_edge_pairs_preserve_legacy_order_and_counts():
    config = SoftBlockConfig()
    pairs = build_edge_pairs(config); metadata = build_edge_metadata(config)
    assert {key: len(value) for key, value in pairs.items()} == {
        "structural": 162, "shear": 242, "bending": 108}
    for kind in pairs:
        assert np.array_equal(pairs[kind], np.asarray(
            [[edge["a"], edge["b"]] for edge in metadata[kind]]))


def test_kelvin_block_has_no_internal_p2p_and_balanced_forces():
    client = bullet_client.BulletClient(pybullet.DIRECT)
    try:
        material = KelvinVoigtMaterial(8, .02, 3, .012, .8, .004)
        block = KelvinVoigtSoftBlock(
            client, SoftBlockConfig(), material, .3, (.4, -.15))
        assert block.constraint_ids == [] and client.getNumConstraints() == 0
        stats = block.apply_internal_forces()
        assert stats.force_evaluation_count == 512
        assert stats.capped_force_count == 0
    finally:
        client.disconnect()
