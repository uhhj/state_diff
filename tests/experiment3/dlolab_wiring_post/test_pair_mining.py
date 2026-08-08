from scripts.experiment3.dlolab_wiring_post.features import (
    hidden_difference,
)


def test_hidden_difference_detects_contact_proxy_mismatch():
    first = [
        {
            "wrap_angle_rad": 0.1,
            "surface_clearance_m": 0.02,
            "contact_like": False,
            "nearest_vertex": 1,
        }
    ]

    second = [
        {
            "wrap_angle_rad": 0.1,
            "surface_clearance_m": 0.0,
            "contact_like": True,
            "nearest_vertex": 1,
        }
    ]

    result = hidden_difference(
        first,
        second,
        0.75,
    )

    assert result[
        "different"
    ] is True

    assert result[
        "contact_like_mismatch"
    ] is True
