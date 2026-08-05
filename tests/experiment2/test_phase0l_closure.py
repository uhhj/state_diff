import inspect
import json
from pathlib import Path

from scripts.experiment2.phase0 import close_hidden_latch_family as closure


ROOT = Path(__file__).resolve().parents[2]


def test_closure_reads_only_historical_summaries_and_runs_no_simulation():
    source = inspect.getsource(closure)
    assert "phase0i" in source and "phase0j" in source and "phase0k" in source
    assert "pybullet" not in source.lower()
    assert "run_exact_counterfactual_pair" not in source
    assert '"new_simulation_performed": False' in source
    assert '"training_performed": False' in source


def test_closure_preserves_verdicts_and_limits_claim():
    expected = {
        "phase0i": "HIDDEN_Z_LATCH_SMOKE_BLOCKED",
        "phase0j": "HIDDEN_WIDE_STOP_SMOKE_BLOCKED",
        "phase0k": "HIDDEN_TENSION_EXTENSION_SMOKE_BLOCKED",
    }
    paths = {
        "phase0i": ROOT / "reports/experiment2/phase0_hidden_hook/phase0i/summary.json",
        "phase0j": ROOT / "reports/experiment2/phase0_hidden_hook/phase0j/summary.json",
        "phase0k": ROOT / "reports/experiment2/phase0_hidden_hook/phase0k/summary.json",
    }
    assert {key: json.loads(path.read_text())["verdict"] for key, path in paths.items()} == expected
    source = inspect.getsource(closure)
    assert "delayed_z_latch family" in source
    assert "whole-cable mean progress gap" in source
    assert "does not reject hidden" in source
