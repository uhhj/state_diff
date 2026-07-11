#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("ravens")
    pkg.__path__ = [str(root / "external/deformable-ravens/ravens")]
    pkg.__package__ = "ravens"
    sys.modules.setdefault("ravens", pkg)
    root = Path(__file__).resolve().parents[1]
    submodule = root / "external/deformable-ravens"
    if str(submodule) not in sys.path:
        sys.path.insert(0, str(submodule))

    from ravens.tasks.ccda_slack_breakaway import (
        DORMANT,
        ENGAGED,
        RELEASED,
        SlackBreakawayConfig,
        UnilateralSlackBreakaway,
    )

    config = SlackBreakawayConfig(
        slack_distance=0.01,
        spring_stiffness=100.0,
        radial_damping=0.2,
        max_tension=4.0,
        breakaway_extension=0.03,
        breakaway_force=3.0,
    )
    model = UnilateralSlackBreakaway(config, [0.0, 0.0, 0.1])

    dormant = model.evaluate(
        [0.005, 0.0, 0.1],
        [100.0, 0.0, 0.0],
        physics_step=1,
    )
    assert dormant.state == DORMANT
    assert dormant.tension == 0.0
    np.testing.assert_array_equal(dormant.force_xyz, np.zeros(3))

    engaged = model.evaluate(
        [0.02, 0.0, 0.1],
        [0.0, 0.0, 0.0],
        physics_step=2,
    )
    assert engaged.state == ENGAGED
    assert engaged.force_xyz[0] < 0.0

    snapshot = copy.deepcopy(model.snapshot())
    restored = UnilateralSlackBreakaway.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot

    released = restored.evaluate(
        [0.041, 0.0, 0.1],
        [0.0, 0.0, 0.0],
        physics_step=3,
    )
    assert released.state == RELEASED
    assert released.tension == 0.0
    np.testing.assert_array_equal(released.force_xyz, np.zeros(3))

    payload = {
        "status": "PASS",
        "dormant_zero_force": True,
        "engaged_pull_direction": True,
        "snapshot_round_trip": True,
        "release_zero_force": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
