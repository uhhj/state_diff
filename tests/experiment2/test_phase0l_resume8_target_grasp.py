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
from scripts.experiment2.phase0.run_hidden_friction_pairs import (
    _environment_action,
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


def test_resume8_actions_target_frozen_beads(
    monkeypatch,
):
    config = load_config(
        "hidden_routing_gate_"
        "phase0l_resume8.json"
    )
    beads = load_beads()
    public = freeze_public(
        monkeypatch,
        config,
        beads,
    )

    actions = (
        common
        .generate_hidden_routing_gate_action_script(
            config,
            beads,
        )
    )

    assert len(actions) == 2
    preload, main = actions

    assert preload[
        "target_grasp_mode"
    ] == "selected_bead_only"
    assert main[
        "target_grasp_mode"
    ] == "selected_bead_only"

    assert preload[
        "target_bead_index"
    ] == int(public["probe_index"])
    assert main[
        "target_bead_index"
    ] == int(public["endpoint_index"])

    assert preload[
        "public_task_layout"
    ] == public
    assert main[
        "public_task_layout"
    ] == public


def test_target_indices_reach_primitives(
    monkeypatch,
):
    config = load_config(
        "hidden_routing_gate_"
        "phase0l_resume8.json"
    )
    beads = load_beads()
    freeze_public(
        monkeypatch,
        config,
        beads,
    )

    preload, main = (
        common
        .generate_hidden_routing_gate_action_script(
            config,
            beads,
        )
    )

    preload_env = (
        _environment_action(preload)
    )
    main_env = (
        _environment_action(main)
    )

    assert preload_env[
        "params"
    ][
        "target_bead_index"
    ] == preload[
        "target_bead_index"
    ]
    assert main_env[
        "params"
    ][
        "target_bead_index"
    ] == main[
        "target_bead_index"
    ]


def test_resume7_default_has_no_target_override(
    monkeypatch,
):
    config = load_config(
        "hidden_routing_gate_"
        "phase0l_resume7.json"
    )
    beads = load_beads()
    freeze_public(
        monkeypatch,
        config,
        beads,
    )

    actions = (
        common
        .generate_hidden_routing_gate_action_script(
            config,
            beads,
        )
    )

    assert all(
        action[
            "target_grasp_mode"
        ] == "generic_contact"
        for action in actions
    )
    assert all(
        "target_bead_index"
        not in action
        for action in actions
    )
