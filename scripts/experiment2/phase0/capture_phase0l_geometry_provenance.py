#!/usr/bin/env python3
"""Capture the original Phase 0L geometry rejection on seed 71001."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = (
    REPO_ROOT / "external" / "deformable-ravens"
)
for path in (REPO_ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    HiddenRoutingGateGeometryError,
    evaluate_hidden_routing_gate_candidates,
)
from scripts.experiment2.phase0.common import (
    canonical_json_sha256,
)
from scripts.experiment2.phase0.exact_counterfactual import (
    run_exact_counterfactual_pair,
)
from scripts.experiment2.phase0.hidden_routing_gate_common import (
    configure_hidden_routing_gate_environment,
    generate_hidden_routing_gate_action_script,
    routing_geometry_config,
)
from scripts.experiment2.phase0.run_hidden_hook_bootstrap import (
    git_sha,
    write_json,
)


def _load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _candidate_widths(audit):
    return sorted({
        round(
            float(
                row[
                    "required_barrier_width"
                ]
            ),
            15,
        )
        for row in audit["candidates"]
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-config",
        default=(
            "configs/experiment2/phase0/"
            "hidden_routing_gate_phase0l.json"
        ),
    )
    parser.add_argument(
        "--resume-config",
        default=(
            "configs/experiment2/phase0/"
            "hidden_routing_gate_phase0l_"
            "resume1.json"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=71001,
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/experiment2/"
            "phase0_hidden_routing_gate/"
            "phase0l_resume1/"
            "geometry_provenance_original.json"
        ),
    )
    args = parser.parse_args()

    if args.seed != 71001:
        raise ValueError(
            "Resume1 provenance is fixed "
            "to seed 71001"
        )

    original_path = (
        REPO_ROOT / args.original_config
    ).resolve()
    resume_path = (
        REPO_ROOT / args.resume_config
    ).resolve()
    output_path = (
        REPO_ROOT / args.output
    ).resolve()

    original = _load(original_path)
    resume = _load(resume_path)

    original_width = float(
        original["routing_gate"][
            "barrier_width"
        ]
    )
    fixed_width = float(
        resume["routing_gate"][
            "barrier_width"
        ]
    )
    if original_width != 0.340:
        raise ValueError(
            "original Phase 0L width changed"
        )
    if fixed_width != 0.350:
        raise ValueError(
            "Resume1 width changed"
        )

    group_id = (
        f"rg_geometry_preflight_"
        f"{args.seed:06d}"
    )
    execution = dict(
        original["execution"]
    )
    observation = dict(
        original["observation"]
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix=(
                "phase0l_geometry_"
                "provenance_"
            )
        ) as temporary:
            run_exact_counterfactual_pair(
                original,
                seed=args.seed,
                group_id=group_id,
                execution=execution,
                observation_output_dir=(
                    Path(temporary)
                    / "observations"
                ),
                observation_config=observation,
                hidden_condition="hidden_hook",
                configure_environment_fn=(
                    configure_hidden_routing_gate_environment
                ),
                action_generator_fn=(
                    generate_hidden_routing_gate_action_script
                ),
            )
    except HiddenRoutingGateGeometryError as error:
        original_audit = (
            error.diagnostics
        )
    else:
        raise RuntimeError(
            "original Phase 0L geometry "
            "unexpectedly became legal"
        )

    beads = original_audit[
        "bead_positions_xyz"
    ]
    fixed_geometry = replace(
        routing_geometry_config(original),
        barrier_width=fixed_width,
        topology_id=str(
            resume["topology_id"]
        ),
    )
    fixed_audit = (
        evaluate_hidden_routing_gate_candidates(
            beads,
            fixed_geometry,
        )
    )

    original_required = (
        _candidate_widths(
            original_audit
        )
    )
    fixed_required = (
        _candidate_widths(
            fixed_audit
        )
    )
    if original_required != fixed_required:
        raise RuntimeError(
            "required width changed when only "
            "configured width changed"
        )
    maximum_required = max(
        original_required
    )

    original_coverage_only = bool(
        original_audit[
            "candidate_count"
        ] == 4
        and original_audit[
            "accepted_candidate_count"
        ] == 0
        and original_audit[
            "rejection_counts"
        ] == {
            "barrier_coverage": 4
        }
    )
    fixed_has_legal_candidate = bool(
        fixed_audit[
            "accepted_candidate_count"
        ] >= 1
    )
    fixed_width_justified = bool(
        original_width
        < maximum_required
        <= fixed_width
    )
    passed = bool(
        original_coverage_only
        and fixed_has_legal_candidate
        and fixed_width_justified
    )

    payload: dict[str, Any] = {
        "stage": (
            "Experiment2 Phase 0L "
            "Resume1 Geometry Provenance"
        ),
        "engineering_only": True,
        "scientific_pair_completed": False,
        "scientific_metrics_computed": False,
        "training_performed": False,
        "geometry_search_performed": False,
        "seed": int(args.seed),
        "group_id": group_id,
        "runtime_main_sha": git_sha(
            REPO_ROOT
        ),
        "runtime_submodule_sha": git_sha(
            SUBMODULE_ROOT
        ),
        "original_config": str(
            original_path.relative_to(
                REPO_ROOT
            )
        ),
        "resume_config": str(
            resume_path.relative_to(
                REPO_ROOT
            )
        ),
        "original_config_hash": (
            canonical_json_sha256(
                original
            )
        ),
        "resume_config_hash": (
            canonical_json_sha256(
                resume
            )
        ),
        "original_width": original_width,
        "fixed_width": fixed_width,
        "maximum_required_width": (
            maximum_required
        ),
        "original_coverage_only": (
            original_coverage_only
        ),
        "fixed_has_legal_candidate": (
            fixed_has_legal_candidate
        ),
        "fixed_width_justified": (
            fixed_width_justified
        ),
        "passed": passed,
        "original_audit": original_audit,
        "fixed_offline_audit": fixed_audit,
    }
    write_json(
        output_path,
        payload,
    )
    if not passed:
        raise RuntimeError(
            "Phase 0L Resume1 geometry "
            "provenance did not match the "
            "pre-registered coverage repair"
        )
    print(output_path)


if __name__ == "__main__":
    main()

