#!/usr/bin/env python3
"""Offline endpoint-corridor geometry preflight from Resume1 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = (
    REPO_ROOT
    / "external"
    / "deformable-ravens"
)
for path in (
    REPO_ROOT,
    SUBMODULE_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )

from ravens.tasks.ccda_hidden_routing_gate_geometry import (
    ENDPOINT_CORRIDOR_BARRIER_MODE,
    compute_hidden_routing_gate_layout,
    evaluate_hidden_routing_gate_candidates,
)
from scripts.experiment2.phase0.common import (
    canonical_json_sha256,
)
from scripts.experiment2.phase0.hidden_routing_gate_common import (
    routing_geometry_config,
)
from scripts.experiment2.phase0.run_hidden_hook_bootstrap import (
    git_sha,
    write_json,
)


def _load(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume1-provenance",
        default=(
            "reports/experiment2/"
            "phase0_hidden_routing_gate/"
            "phase0l_resume1/"
            "geometry_provenance_original.json"
        ),
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/experiment2/phase0/"
            "hidden_routing_gate_"
            "phase0l_resume2.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/experiment2/"
            "phase0_hidden_routing_gate/"
            "phase0l_resume2/"
            "corridor_geometry_preflight.json"
        ),
    )
    args = parser.parse_args()

    provenance_path = (
        REPO_ROOT
        / args.resume1_provenance
    ).resolve()
    config_path = (
        REPO_ROOT / args.config
    ).resolve()
    output_path = (
        REPO_ROOT / args.output
    ).resolve()

    provenance = _load(
        provenance_path
    )
    config = _load(
        config_path
    )

    expected_sha = str(
        config["geometry_repair"][
            "source_resume1_provenance_"
            "sha256"
        ]
    )
    actual_sha = _sha256(
        provenance_path
    )
    if actual_sha != expected_sha:
        raise RuntimeError(
            "Resume1 provenance hash "
            "does not match Resume2 config"
        )

    original = provenance["original_audit"]
    if (
        int(
            original[
                "candidate_count"
            ]
        )
        != 4
    ):
        raise RuntimeError(
            "Resume1 candidate count changed"
        )
    if original[
        "rejection_counts"
    ] != {
        "workspace": 4
    }:
        raise RuntimeError(
            "Resume1 root cause is not "
            "workspace-only"
        )

    geometry = (
        routing_geometry_config(
            config
        )
    )
    if str(
        geometry.barrier_mode
    ) != (
        ENDPOINT_CORRIDOR_BARRIER_MODE
    ):
        raise RuntimeError(
            "Resume2 must use "
            "endpoint-corridor mode"
        )

    beads = original[
        "bead_positions_xyz"
    ]
    audit = (
        evaluate_hidden_routing_gate_candidates(
            beads,
            geometry,
        )
    )
    layout = (
        compute_hidden_routing_gate_layout(
            beads,
            geometry,
        )
    )

    accepted = [
        row
        for row in audit["candidates"]
        if row["accepted"]
    ]
    rejected = [
        row
        for row in audit["candidates"]
        if not row["accepted"]
    ]

    resolved_width = float(
        geometry.resolved_barrier_width
    )
    expected_width = float(
        config["geometry_repair"][
            "resolved_barrier_width"
        ]
    )
    width_match = bool(
        abs(
            resolved_width
            - expected_width
        )
        <= 1e-15
    )
    accepted_count_ok = bool(
        len(accepted) >= 1
    )
    accepted_geometry_ok = bool(
        accepted
        and all(
            row[
                "coverage_margin"
            ] >= (
                geometry
                .barrier_safety_margin
                - 1e-12
            )
            and row[
                "workspace_margin"
            ] > 0
            and row[
                "minimum_clearance"
            ] + 1e-9
            >= row[
                "expected_minimum_clearance"
            ]
            and abs(
                row[
                    "barrier_tangent_center_error"
                ]
            ) <= 1e-12
            for row in accepted
        )
    )
    no_parameter_search = bool(
        config[
            "topology_policy"
        ]
        == (
            "single_fixed_topology_"
            "no_grid_search"
        )
        and "candidates"
        not in config
    )

    passed = bool(
        width_match
        and accepted_count_ok
        and accepted_geometry_ok
        and no_parameter_search
    )

    selected_public = dict(
        layout["public_task"]
    )
    selected = {
        "endpoint_index": int(
            layout["endpoint_index"]
        ),
        "normal_sign": float(
            layout["normal_sign"]
        ),
        "barrier_mode": str(
            layout["barrier_mode"]
        ),
        "resolved_barrier_width": (
            float(
                layout[
                    "resolved_barrier_width"
                ]
            )
        ),
        "coverage_margin": float(
            layout[
                "barrier_tangent_"
                "coverage_margin"
            ]
        ),
        "workspace_margin": float(
            layout["workspace_margin"]
        ),
        "component_workspace_margin": (
            layout[
                "component_workspace_margin"
            ]
        ),
        "barrier_surface_clearance": (
            float(
                layout[
                    "barrier_surface_clearance"
                ]
            )
        ),
        "probe_roof_surface_clearance": (
            float(
                layout[
                    "probe_roof_surface_clearance"
                ]
            )
        ),
        "public_task_layout_sha256": (
            canonical_json_sha256(
                selected_public
            )
        ),
    }

    payload = {
        "stage": (
            "Experiment2 Phase 0L "
            "Resume2 Geometry Preflight"
        ),
        "engineering_only": True,
        "source_resume1_provenance": str(
            provenance_path.relative_to(
                REPO_ROOT
            )
        ),
        "source_resume1_provenance_sha256": (
            actual_sha
        ),
        "source_seed": int(
            provenance["seed"]
        ),
        "runtime_main_sha": git_sha(
            REPO_ROOT
        ),
        "runtime_submodule_sha": git_sha(
            SUBMODULE_ROOT
        ),
        "config": str(
            config_path.relative_to(
                REPO_ROOT
            )
        ),
        "config_hash": (
            canonical_json_sha256(
                config
            )
        ),
        "barrier_mode": str(
            geometry.barrier_mode
        ),
        "barrier_width_source": (
            "derived_formula"
        ),
        "resolved_barrier_width": (
            resolved_width
        ),
        "corridor_required_width": (
            float(
                geometry
                .corridor_required_width
            )
        ),
        "barrier_safety_margin": (
            float(
                geometry
                .barrier_safety_margin
            )
        ),
        "candidate_count": int(
            audit["candidate_count"]
        ),
        "accepted_candidate_count": (
            int(
                audit[
                    "accepted_candidate_count"
                ]
            )
        ),
        "rejection_counts": (
            audit["rejection_counts"]
        ),
        "accepted_candidates": (
            accepted
        ),
        "rejected_candidates": (
            rejected
        ),
        "selected_candidate": selected,
        "width_match": width_match,
        "accepted_count_ok": (
            accepted_count_ok
        ),
        "accepted_geometry_ok": (
            accepted_geometry_ok
        ),
        "no_parameter_search": (
            no_parameter_search
        ),
        "scientific_pair_completed": (
            False
        ),
        "scientific_metrics_computed": (
            False
        ),
        "training_performed": False,
        "geometry_search_performed": (
            False
        ),
        "action_search_performed": (
            False
        ),
        "outcome_search_performed": (
            False
        ),
        "passed": passed,
    }
    write_json(
        output_path,
        payload,
    )
    if not passed:
        raise RuntimeError(
            "Phase 0L Resume2 "
            "corridor geometry preflight "
            "failed"
        )
    print(output_path)


if __name__ == "__main__":
    main()
