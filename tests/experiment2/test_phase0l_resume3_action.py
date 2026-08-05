import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_ROOT = ROOT / "external" / "deformable-ravens"
for path in (ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.experiment2.phase0.hidden_routing_gate_common import (
    generate_hidden_routing_gate_action_script,
)


CONFIG = (
    ROOT
    / "configs"
    / "experiment2"
    / "phase0"
    / (
        "hidden_routing_gate_"
        "phase0l_resume3.json"
    )
)
FIXTURE = (
    ROOT
    / "external"
    / "deformable-ravens"
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


def test_action_uses_selected_probe_override(
    monkeypatch,
):
    monkeypatch.setenv(
        "CCDA_ROUTING_SELECTED_PROBE_INDEX",
        "7",
    )
    actions = (
        generate_hidden_routing_gate_action_script(
            load_config(),
            load_beads(),
        )
    )
    assert len(actions) == 2
    assert [
        action["phase"]
        for action in actions
    ] == [
        "preload",
        "main_pull",
    ]
    for action in actions:
        assert (
            action[
                "public_task_layout"
            ][
                "probe_index"
            ]
            == 7
        )


def test_resume3_action_keeps_existing_primitives(
    monkeypatch,
):
    monkeypatch.setenv(
        "CCDA_ROUTING_SELECTED_PROBE_INDEX",
        "7",
    )
    actions = (
        generate_hidden_routing_gate_action_script(
            load_config(),
            load_beads(),
        )
    )
    assert actions[0]["primitive"] == (
        "pick_precise_latch_probe"
    )
    assert actions[1]["primitive"] == (
        "pick_precise_tension_extension"
    )
