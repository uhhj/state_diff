import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_ROOT = (
    ROOT
    / "external"
    / "deformable-ravens"
)
for path in (
    ROOT,
    SUBMODULE_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )

from ravens.tasks.ccda_hidden_routing_gate_cable import (
    FROZEN_PUBLIC_LAYOUT_ENV,
)
from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    compute_hidden_routing_gate_layout,
    public_routing_layout,
)
from scripts.experiment2.phase0 import (
    hidden_routing_gate_common
    as common,
)


CONFIGS = (
    ROOT
    / "configs"
    / "experiment2"
    / "phase0"
)
FIXTURE = (
    SUBMODULE_ROOT
    / "tests"
    / "data"
    / (
        "ccda_hidden_routing_gate_"
        "seed71001_settled_beads.json"
    )
)


def load_config(name):
    return json.loads(
        (
            CONFIGS / name
        ).read_text(
            encoding="utf-8"
        )
    )


def load_beads():
    return np.asarray(
        json.loads(
            FIXTURE.read_text(
                encoding="utf-8"
            )
        ),
        dtype=np.float64,
    )


def freeze_public(
    monkeypatch,
    config,
    beads,
):
    layout = (
        compute_hidden_routing_gate_layout(
            beads,
            common.routing_geometry_config(
                config
            ),
        )
    )
    public = public_routing_layout(
        layout
    )
    monkeypatch.setenv(
        FROZEN_PUBLIC_LAYOUT_ENV,
        json.dumps(
            public,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return public


def main_action(actions):
    matches = [
        action
        for action in actions
        if action["phase"] == "main_pull"
    ]
    assert len(matches) == 1
    return matches[0]


def unit(vector):
    value = np.asarray(
        vector,
        dtype=np.float64,
    )
    return value / np.linalg.norm(value)


def test_resume5_relative_pull_is_exactly_collinear(
    monkeypatch,
):
    config = load_config(
        "hidden_routing_gate_"
        "phase0l_resume5.json"
    )
    base = load_beads()
    public = freeze_public(
        monkeypatch,
        config,
        base,
    )

    tangent = unit(
        public["tangent_xy"]
    )
    stable = base.copy()
    stable[:, :2] += (
        0.0004 * tangent
    )

    actions = (
        common
        .generate_hidden_routing_gate_action_script(
            config,
            stable,
        )
    )
    action = main_action(actions)

    pose0 = np.asarray(
        action["pose0"]["position"],
        dtype=np.float64,
    )
    stage1 = np.asarray(
        action[
            "pose_stage1"
        ]["position"],
        dtype=np.float64,
    )
    final = np.asarray(
        action["pose1"]["position"],
        dtype=np.float64,
    )

    stage1_vector = (
        stage1[:2] - pose0[:2]
    )
    final_vector = (
        final[:2] - pose0[:2]
    )

    assert action[
        "main_pull_frame"
    ] == (
        "current_endpoint_frozen_normal"
    )
    assert np.isclose(
        np.linalg.norm(
            stage1_vector
        ),
        0.080,
        atol=1e-12,
    )
    assert np.isclose(
        np.linalg.norm(
            final_vector
        ),
        0.120,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        unit(stage1_vector),
        unit(final_vector),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        unit(stage1_vector),
        unit(public["normal_xy"]),
        atol=1e-12,
    )


def test_resume5_uses_current_endpoint_pose0(
    monkeypatch,
):
    config = load_config(
        "hidden_routing_gate_"
        "phase0l_resume5.json"
    )
    base = load_beads()
    public = freeze_public(
        monkeypatch,
        config,
        base,
    )

    tangent = unit(
        public["tangent_xy"]
    )
    stable = base.copy()
    stable[:, :2] += (
        0.0004 * tangent
    )

    actions = (
        common
        .generate_hidden_routing_gate_action_script(
            config,
            stable,
        )
    )
    action = main_action(actions)

    endpoint_index = int(
        public["endpoint_index"]
    )
    pose0 = np.asarray(
        action["pose0"]["position"],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        pose0[:2],
        stable[
            endpoint_index,
            :2,
        ],
        atol=1e-15,
    )


def test_resume5_keeps_frozen_task_layout(
    monkeypatch,
):
    config = load_config(
        "hidden_routing_gate_"
        "phase0l_resume5.json"
    )
    base = load_beads()
    public = freeze_public(
        monkeypatch,
        config,
        base,
    )

    stable = base.copy()
    stable[:, 0] += 0.0003
    stable[:, 1] -= 0.0002

    actions = (
        common
        .generate_hidden_routing_gate_action_script(
            config,
            stable,
        )
    )

    assert all(
        action[
            "public_task_layout"
        ] == public
        for action in actions
    )

    action_public = main_action(actions)[
        "public_task_layout"
    ]
    assert action_public[
        "target_plane_point_xy"
    ] == public[
        "target_plane_point_xy"
    ]
    assert action_public[
        "target_zone_center_xy"
    ] == public[
        "target_zone_center_xy"
    ]
    assert action_public[
        "target_corridor_half_width"
    ] == public[
        "target_corridor_half_width"
    ]


def test_resume5_targets_are_relative_not_nominal_absolute(
    monkeypatch,
):
    config = load_config(
        "hidden_routing_gate_"
        "phase0l_resume5.json"
    )
    base = load_beads()
    public = freeze_public(
        monkeypatch,
        config,
        base,
    )

    tangent = unit(
        public["tangent_xy"]
    )
    stable = base.copy()
    stable[:, :2] += (
        0.0004 * tangent
    )

    action = main_action(
        common
        .generate_hidden_routing_gate_action_script(
            config,
            stable,
        )
    )

    commanded_stage1 = np.asarray(
        action[
            "pose_stage1"
        ]["position"][:2],
        dtype=np.float64,
    )
    nominal_stage1 = np.asarray(
        public["stage1_target_xy"],
        dtype=np.float64,
    )

    assert not np.allclose(
        commanded_stage1,
        nominal_stage1,
        atol=1e-10,
    )


def test_resume4_default_absolute_behavior_is_preserved(
    monkeypatch,
):
    config = load_config(
        "hidden_routing_gate_"
        "phase0l_resume4.json"
    )
    base = load_beads()
    public = freeze_public(
        monkeypatch,
        config,
        base,
    )

    stable = base.copy()
    stable[:, 0] += 0.0003

    action = main_action(
        common
        .generate_hidden_routing_gate_action_script(
            config,
            stable,
        )
    )

    np.testing.assert_allclose(
        action[
            "pose_stage1"
        ]["position"][:2],
        public["stage1_target_xy"],
        atol=1e-15,
    )
    np.testing.assert_allclose(
        action[
            "pose1"
        ]["position"][:2],
        public["final_target_xy"],
        atol=1e-15,
    )
