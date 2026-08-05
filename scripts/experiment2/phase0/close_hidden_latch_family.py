#!/usr/bin/env python3
"""Create an immutable scientific closure summary for Phase 0I/J/K."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase0i",
        default=(
            "reports/experiment2/"
            "phase0_hidden_hook/phase0i/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--phase0j",
        default=(
            "reports/experiment2/"
            "phase0_hidden_hook/phase0j/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--phase0k",
        default=(
            "reports/experiment2/"
            "phase0_hidden_hook/phase0k/"
            "summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/experiment2/"
            "phase0_hidden_routing_gate/"
            "phase0l/"
            "hidden_latch_family_closure"
        ),
    )
    args = parser.parse_args()

    paths = {
        "phase0i": (
            REPO_ROOT / args.phase0i
        ).resolve(),
        "phase0j": (
            REPO_ROOT / args.phase0j
        ).resolve(),
        "phase0k": (
            REPO_ROOT / args.phase0k
        ).resolve(),
    }
    reports = {
        key: load_json(value)
        for key, value in paths.items()
    }

    expected = {
        "phase0i": (
            "HIDDEN_Z_LATCH_SMOKE_BLOCKED"
        ),
        "phase0j": (
            "HIDDEN_WIDE_STOP_SMOKE_BLOCKED"
        ),
        "phase0k": (
            "HIDDEN_TENSION_EXTENSION_"
            "SMOKE_BLOCKED"
        ),
    }
    for key, verdict in expected.items():
        actual = str(
            reports[key].get("verdict")
        )
        if actual != verdict:
            raise RuntimeError(
                f"{key} verdict changed: "
                f"{actual!r} != {verdict!r}"
            )

    closure = {
        "stage": (
            "Experiment2 Phase 0L "
            "Hidden Z-latch Family Closure"
        ),
        "post_hoc_summary_only": True,
        "historical_verdicts_changed": False,
        "new_simulation_performed": False,
        "training_performed": False,
        "closed_task_mechanism": (
            "delayed_z_latch family"
        ),
        "closed_official_outcome": (
            "whole-cable mean progress gap"
        ),
        "closure_reason": (
            "Phase 0I established an observable "
            "future-state branch but insufficient "
            "positive mean progress. Phase 0J "
            "degraded the Pareto metrics when the "
            "barrier was widened. Phase 0K retained "
            "the branch but produced seed-dependent "
            "and negative signed progress under "
            "additional same-end tension."
        ),
        "scope_limit": (
            "This closes only the delayed Z-latch "
            "plus whole-cable mean-progress task "
            "combination. It does not reject hidden "
            "contact conditioning or outcome-aligned "
            "routing tasks."
        ),
        "source_reports": {
            key: str(
                path.relative_to(REPO_ROOT)
            )
            for key, path in paths.items()
        },
        "source_verdicts": {
            key: reports[key]["verdict"]
            for key in reports
        },
        "next_task_family": (
            "hidden_routing_gate"
        ),
    }

    output = (
        REPO_ROOT / args.output
    ).resolve()
    write_json(
        output / "closure.json",
        closure,
    )
    lines = [
        "# Hidden Z-latch Family Closure",
        "",
        "- Post-hoc summary only: `True`",
        "- Historical verdicts changed: `False`",
        "- New simulation performed: `False`",
        "- Training performed: `False`",
        "- Closed combination: "
        "`delayed Z-latch + whole-cable "
        "mean progress gap`",
        "- Phase 0I verdict: "
        f"`{reports['phase0i']['verdict']}`",
        "- Phase 0J verdict: "
        f"`{reports['phase0j']['verdict']}`",
        "- Phase 0K verdict: "
        f"`{reports['phase0k']['verdict']}`",
        "- Next task family: "
        "`hidden_routing_gate`",
        "",
        "This closure does not reject CCDA or "
        "hidden-contact conditioning. It closes "
        "only the prior task-mechanism-outcome "
        "combination.",
    ]
    (
        output / "summary.md"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(output / "closure.json")


if __name__ == "__main__":
    main()
