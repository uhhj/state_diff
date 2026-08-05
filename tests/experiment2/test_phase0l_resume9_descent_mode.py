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


def test_resume9_actions_use_single_pass_descent(
    monkeypatch,
):
    config = load_config(
        "hidden_routing_gate_"
        "phase0l_resume9.json"
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

    assert len(actions) == 2
    assert all(
        action[
            "acquisition_descent_mode"
        ] == "single_pass_target"
        for action in actions
    )
    assert all(
        action[
            "target_grasp_mode"
        ] == "selected_bead_only"
        for action in actions
    )
    assert all(
        "target_bead_index"
        in action
        for action in actions
    )


def test_descent_mode_reaches_both_primitives(
    monkeypatch,
):
    config = load_config(
        "hidden_routing_gate_"
        "phase0l_resume9.json"
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
        "acquisition_descent_mode"
    ] == "single_pass_target"
    assert main_env[
        "params"
    ][
        "acquisition_descent_mode"
    ] == "single_pass_target"


def test_resume8_defaults_to_stepwise_descent(
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

    actions = (
        common
        .generate_hidden_routing_gate_action_script(
            config,
            beads,
        )
    )

    assert all(
        action[
            "acquisition_descent_mode"
        ] == "legacy_stepwise"
        for action in actions
    )

    env_actions = [
        _environment_action(action)
        for action in actions
    ]
    assert all(
        value["params"][
            "acquisition_descent_mode"
        ] == "legacy_stepwise"
        for value in env_actions
    )
