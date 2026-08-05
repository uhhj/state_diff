import json
import os
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


CONFIG = (
    ROOT
    / "configs"
    / "experiment2"
    / "phase0"
    / (
        "hidden_routing_gate_"
        "phase0l_resume4.json"
    )
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


def load_config():
    return json.loads(
        CONFIG.read_text(
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


def frozen_public(config, beads):
    layout = (
        compute_hidden_routing_gate_layout(
            beads,
            common.routing_geometry_config(
                config
            ),
        )
    )
    return public_routing_layout(
        layout
    )


def test_action_reuses_frozen_public_layout(
    monkeypatch,
):
    config = load_config()
    base = load_beads()
    public = frozen_public(
        config,
        base,
    )
    monkeypatch.setenv(
        FROZEN_PUBLIC_LAYOUT_ENV,
        json.dumps(
            public,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )

    stable = base.copy()
    stable[:, 0] += 0.0004
    stable[:, 1] -= 0.0002

    actions = (
        common
        .generate_hidden_routing_gate_action_script(
            config,
            stable,
        )
    )

    assert len(actions) == 2
    assert all(
        action[
            "public_task_layout"
        ] == public
        for action in actions
    )

    probe_index = int(
        public["probe_index"]
    )
    endpoint_index = int(
        public["endpoint_index"]
    )

    probe_pose = np.asarray(
        actions[0]["pose0"]["position"],
        dtype=np.float64,
    )
    main_pose = np.asarray(
        actions[1]["pose0"]["position"],
        dtype=np.float64,
    )

    assert np.allclose(
        probe_pose[:2],
        stable[
            probe_index,
            :2,
        ],
    )
    assert np.allclose(
        main_pose[:2],
        stable[
            endpoint_index,
            :2,
        ],
    )
    assert np.allclose(
        actions[1][
            "pose_stage1"
        ]["position"][:2],
        public[
            "stage1_target_xy"
        ],
    )
    assert np.allclose(
        actions[1][
            "pose1"
        ]["position"][:2],
        public[
            "final_target_xy"
        ],
    )


def test_frozen_mode_does_not_recompute_layout(
    monkeypatch,
):
    config = load_config()
    beads = load_beads()
    public = frozen_public(
        config,
        beads,
    )
    monkeypatch.setenv(
        FROZEN_PUBLIC_LAYOUT_ENV,
        json.dumps(public),
    )

    def fail(*args, **kwargs):
        raise AssertionError(
            "layout constructor was called "
            "during frozen action generation"
        )

    monkeypatch.setattr(
        common,
        "compute_hidden_routing_gate_layout",
        fail,
    )

    actions = (
        common
        .generate_hidden_routing_gate_action_script(
            config,
            beads,
        )
    )
    assert actions[0][
        "public_task_layout"
    ] == public


def test_environment_clears_stale_frozen_layout(
    monkeypatch,
):
    config = load_config()
    monkeypatch.setenv(
        FROZEN_PUBLIC_LAYOUT_ENV,
        "{\"stale\":true}",
    )

    common.configure_hidden_routing_gate_environment(
        config,
        "free",
        71001,
        "rg_071001",
    )

    assert (
        FROZEN_PUBLIC_LAYOUT_ENV
        not in os.environ
    )
