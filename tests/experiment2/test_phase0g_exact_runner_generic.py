import pytest

from scripts.experiment2.phase0.exact_counterfactual import _arm_ccda_hidden_factor


def test_generic_arm_method_has_priority():
    calls = []

    class Task:
        def arm_ccda_hidden_factor_after_settle(self):
            calls.append("generic")

        def arm_hidden_friction_after_settle(self):
            calls.append("legacy")

    _arm_ccda_hidden_factor(Task())
    assert calls == ["generic"]


def test_legacy_friction_arm_fallback_is_preserved():
    calls = []

    class Task:
        def arm_hidden_friction_after_settle(self):
            calls.append("legacy")

    _arm_ccda_hidden_factor(Task())
    assert calls == ["legacy"]


def test_missing_arm_method_is_rejected():
    with pytest.raises(AttributeError):
        _arm_ccda_hidden_factor(object())
