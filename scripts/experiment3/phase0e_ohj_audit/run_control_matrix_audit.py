"""Run the frozen R13 action matrix as a diagnostic-only audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_ohj_cable.common import (  # noqa: E402
    write_json)
from scripts.experiment3.phase0_ohj_cable.run_control_relevance import (  # noqa: E402
    collect_control_matrix)


def run_audit(audit_config_path):
    audit = json.loads(Path(audit_config_path).read_text(encoding="utf-8"))
    source_config = REPO_ROOT / audit["source_config"]
    payload = collect_control_matrix(
        source_config, require_pair_complete=False)
    expected = audit["source_pair_verdict_expected"]
    if payload["source_pair_verdict"] != expected:
        raise RuntimeError(
            "unexpected source pair verdict: {}".format(
                payload["source_pair_verdict"]))
    output = Path(audit["raw_output_root"]) / "CONTROL_MATRIX_AUDIT.json"
    result = {
        "diagnostic_only": True,
        "formal_gate5_executed": False,
        "source_config": audit["source_config"],
        "source_pair_verdict": payload["source_pair_verdict"],
        "branch_local_ik": payload["branch_local_ik"],
        "matrix": payload["matrix"],
    }
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_audit(args.config)
    print("source_pair_verdict={}".format(
        result["source_pair_verdict"]))


if __name__ == "__main__":
    main()
