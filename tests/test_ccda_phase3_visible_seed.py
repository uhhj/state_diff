import pytest

from ccda_phase3.data_io import visible_seed_from_extras


@pytest.mark.parametrize(
    ("extras", "expected"),
    [
        ({"ccda_visible_seed": 312000}, 312000),
        ({"ccda_visible_seed": "312500"}, 312500),
        ({"ccda_pair_group": "phase3_12c_seed_312000"}, 312000),
        ({"ccda_pair_group": "phase3_12d_r22_seed_313127"}, 313127),
        ({"ccda_pair_group": "phase3_12c_312000"}, -1),
        ({"ccda_pair_group": "phase3_12c_seed_312000_extra"}, -1),
        (
            {"ccda_visible_seed": "bad", "ccda_pair_group": "phase3_12c_seed_312000"},
            312000,
        ),
        ({}, -1),
    ],
)
def test_visible_seed_parser(extras, expected):
    assert visible_seed_from_extras(extras) == expected
