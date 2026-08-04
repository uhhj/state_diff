import numpy as np
import pytest

from scripts.experiment2.phase0.observation_common import (
    delta_image_feature,
    grouped_ridge_accuracy,
    image_feature,
)


def synthetic_samples(groups=("alpha", "beta", "gamma")):
    samples = []
    for group in groups:
        samples.extend(
            [
                {"group_id": group, "condition": "free", "label": -1, "contact": np.asarray([-2.0, -1.0])},
                {"group_id": group, "condition": "hidden", "label": 1, "contact": np.asarray([2.0, 1.0])},
            ]
        )
    return samples


def test_image_features_have_fixed_length_and_identical_delta_is_zero():
    image = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)
    feature = image_feature(image, width=5, height=4)
    delta = delta_image_feature(image, image.copy(), width=5, height=4)
    assert feature.shape == (5 * 4 * 3,)
    assert delta.shape == feature.shape
    np.testing.assert_array_equal(delta, np.zeros_like(delta))


def test_grouped_ridge_classifies_synthetic_contact_without_using_group_id():
    result = grouped_ridge_accuracy(synthetic_samples(), "contact", l2=0.01)
    assert result["accuracy"] == 1.0
    assert result["feature_key"] == "contact"
    assert result["groups"] == ["alpha", "beta", "gamma"]


def test_grouped_ridge_rejects_fewer_than_three_groups():
    with pytest.raises(ValueError, match="at least three groups"):
        grouped_ridge_accuracy(synthetic_samples(("only", "two")), "contact", l2=0.01)
