import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_ROOT = ROOT / "external" / "deformable-ravens"
for path in (ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.experiment2.phase0.hidden_routing_gate_common import (
    configure_hidden_routing_gate_environment,
    generate_hidden_routing_gate_action_script,
)


def config():
    return json.loads((ROOT / "configs/experiment2/phase0/hidden_routing_gate_phase0l.json").read_text())


def beads():
    return np.column_stack((np.linspace(0.35, 0.65, 25), np.zeros(25), np.full(25, 0.005)))


def actions():
    return generate_hidden_routing_gate_action_script(config(), beads())


def test_action_contains_one_probe_and_one_main():
    result = actions()
    assert [row["phase"] for row in result] == ["preload", "main_pull"]
    assert [row["primitive"] for row in result] == [
        "pick_precise_latch_probe", "pick_precise_tension_extension"
    ]


def test_action_public_layout_has_no_hidden_fields():
    for action in actions():
        serialized = json.dumps(action["public_task_layout"]).lower()
        assert all(word not in serialized for word in ("barrier", "roof", "boxes"))


def test_fixed_distances_are_collinear():
    main = actions()[1]
    start = np.asarray(main["pose0"]["position"][:2])
    stage1 = np.asarray(main["pose_stage1"]["position"][:2])
    final = np.asarray(main["pose1"]["position"][:2])
    assert np.isclose(np.linalg.norm(stage1 - start), 0.080)
    assert np.isclose(np.linalg.norm(final - start), 0.120)
    assert np.allclose((stage1 - start) / 0.080, (final - start) / 0.120)


def test_action_is_identical_for_free_and_hidden_and_endpoint_matches_layout():
    first = actions()
    second = actions()
    assert first == second
    main = first[1]
    public = main["public_task_layout"]
    assert np.allclose(main["pose0"]["position"][:2], beads()[public["endpoint_index"], :2])


def test_configure_clears_stale_selected_probe(
    monkeypatch,
):
    monkeypatch.setenv(
        "CCDA_ROUTING_SELECTED_PROBE_INDEX",
        "20",
    )
    configure_hidden_routing_gate_environment(
        config(),
        "free",
        71001,
        "rg_071001",
    )
    assert (
        "CCDA_ROUTING_SELECTED_PROBE_INDEX"
        not in os.environ
    )
