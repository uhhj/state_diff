from __future__ import annotations

import copy

import sys
import types
from pathlib import Path

import numpy as np

# coord_bimanual intentionally omits TensorFlow. Load the task helper through
# the established TensorFlow-free Ravens namespace used by rollout audits.
_root = Path(__file__).resolve().parents[1]
_pkg = types.ModuleType("ravens")
_pkg.__path__ = [str(_root / "external/deformable-ravens/ravens")]
_pkg.__package__ = "ravens"
sys.modules.setdefault("ravens", _pkg)
import pytest

from ravens.tasks.ccda_slack_breakaway import (
    DORMANT,
    ENGAGED,
    RELEASED,
    SlackBreakawayConfig,
    UnilateralSlackBreakaway,
)


def config() -> SlackBreakawayConfig:
    return SlackBreakawayConfig(
        slack_distance=0.01,
        spring_stiffness=100.0,
        radial_damping=0.2,
        max_tension=4.0,
        breakaway_extension=0.03,
        breakaway_force=3.0,
    )


def test_dormant_region_is_exact_zero_force():
    model = UnilateralSlackBreakaway(
        config(),
        anchor_position=[0.0, 0.0, 0.1],
    )
    output = model.evaluate(
        [0.005, 0.0, 0.1],
        [100.0, 0.0, 0.0],
        physics_step=1,
    )
    assert output.state == DORMANT
    assert output.tension == 0.0
    np.testing.assert_array_equal(output.force_xyz, np.zeros(3))
    assert model.engagement_physics_step is None


def test_force_is_unilateral_and_pulls_toward_anchor():
    model = UnilateralSlackBreakaway(config(), [0.0, 0.0])
    output = model.evaluate(
        [0.02, 0.0],
        [0.1, 0.0],
        physics_step=4,
    )
    assert output.state == ENGAGED
    assert output.engaged_now
    assert output.tension > 0.0
    assert output.force_xyz[0] < 0.0
    assert output.force_xyz[1] == 0.0
    assert output.force_xyz[2] == 0.0


def test_inward_velocity_does_not_create_negative_damping():
    model = UnilateralSlackBreakaway(config(), [0.0, 0.0])
    outward = model.evaluate(
        [0.02, 0.0],
        [0.1, 0.0],
        physics_step=1,
    )
    model = UnilateralSlackBreakaway(config(), [0.0, 0.0])
    inward = model.evaluate(
        [0.02, 0.0],
        [-0.1, 0.0],
        physics_step=1,
    )
    assert outward.tension > inward.tension
    assert inward.tension == pytest.approx(1.0)


def test_release_by_extension_returns_zero_force_immediately():
    model = UnilateralSlackBreakaway(config(), [0.0, 0.0])
    output = model.evaluate(
        [0.041, 0.0],
        [0.0, 0.0],
        physics_step=9,
    )
    assert output.state == RELEASED
    assert output.released_now
    assert output.release_reason == "extension"
    assert output.tension == 0.0
    np.testing.assert_array_equal(output.force_xyz, np.zeros(3))

    later = model.evaluate(
        [0.1, 0.0],
        [10.0, 0.0],
        physics_step=10,
    )
    assert later.state == RELEASED
    assert later.tension == 0.0
    np.testing.assert_array_equal(later.force_xyz, np.zeros(3))


def test_release_by_force():
    cfg = SlackBreakawayConfig(
        slack_distance=0.01,
        spring_stiffness=100.0,
        radial_damping=1.0,
        max_tension=4.0,
        breakaway_extension=0.2,
        breakaway_force=2.0,
    )
    model = UnilateralSlackBreakaway(cfg, [0.0, 0.0])
    output = model.evaluate(
        [0.02, 0.0],
        [2.0, 0.0],
        physics_step=3,
    )
    assert output.state == RELEASED
    assert output.release_reason == "force"
    assert output.tension == 0.0


def test_snapshot_round_trip_is_exact():
    model = UnilateralSlackBreakaway(config(), [0.4, -0.1, 0.02])
    first = model.evaluate(
        [0.42, -0.1, 0.02],
        [0.05, 0.0, 0.0],
        physics_step=12,
    )
    assert first.state == ENGAGED

    snapshot = copy.deepcopy(model.snapshot())
    restored = UnilateralSlackBreakaway.from_snapshot(snapshot)

    np.testing.assert_array_equal(
        restored.last_force_xyz,
        model.last_force_xyz,
    )
    assert restored.snapshot() == model.snapshot()

    second_a = model.evaluate(
        [0.425, -0.1],
        [0.02, 0.0],
        physics_step=13,
    )
    second_b = restored.evaluate(
        [0.425, -0.1],
        [0.02, 0.0],
        physics_step=13,
    )
    np.testing.assert_array_equal(
        second_a.force_xyz,
        second_b.force_xyz,
    )
    assert second_a.state == second_b.state


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slack_distance": 0.0},
        {"spring_stiffness": 0.0},
        {"radial_damping": -1.0},
        {"max_tension": 0.0},
        {"breakaway_extension": 0.0},
        {"breakaway_force": 5.0, "max_tension": 4.0},
    ],
)
def test_invalid_config_rejected(kwargs):
    values = dict(
        slack_distance=0.01,
        spring_stiffness=100.0,
        radial_damping=0.2,
        max_tension=4.0,
        breakaway_extension=0.03,
        breakaway_force=3.0,
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        SlackBreakawayConfig(**values)
