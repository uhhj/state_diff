from collections import Counter

from state_diff.env.block_pushing.soft_block_lattice import (
    SoftBlockConfig, build_angle_triplets)


def test_angle_triplet_count_plane_order_and_uniqueness():
    rows = build_angle_triplets(SoftBlockConfig())
    assert len(rows) == 121
    assert Counter(row.plane for row in rows) == {"xy": 45, "xz": 40, "yz": 36}
    assert [row.plane for row in rows] == ["xy"] * 45 + ["xz"] * 40 + ["yz"] * 36
    assert len(set(rows)) == 121
