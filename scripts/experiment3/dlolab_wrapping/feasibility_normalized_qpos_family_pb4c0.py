"""PB4-C0 — CPU-only feasibility-normalized signed qpos-target family.

Purpose
-------
Derive one globally feasible symmetric qpos-target scale from:
- the frozen PB3-B1/PB4 cohort branch times;
- the verified official Wrapping best_qpos trajectory;
- the pinned Panda hard URDF position limits.

No reward, regret, PB3 future divergence, winding label, sensor, Genesis step,
or GPU execution is used to derive alpha.

The rule is frozen before derivation:
    alpha_formal =
        0.95 * min(alpha_max_hard_limits, 1.0)

where for every formal branch time t, relative step r=1..20, arm block and qpos
DOF with nonzero nominal delta d,

    alpha_limit =
        min(q_t - lower, upper - q_t) / abs(d)

and alpha_max_hard_limits is the global minimum alpha_limit.

This phase never runs PB4-C GPU evaluation. A successful derivation freezes the
family and stops. PB4-C1, if pursued, must be a separate preregistered phase.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from scripts.experiment3.dlolab_wrapping.paths import DLO_ROOT, REPO_ROOT
from scripts.experiment3.dlolab_wrapping.replay_winding_audit import load_json
import scripts.experiment3.dlolab_wrapping.control_relevance_pb4 as pb4


RULE_VERDICT = "PB4C0_FEASIBILITY_RULE_PREREGISTERED"
SUCCESS_VERDICT = "PB4C0_FEASIBILITY_NORMALIZED_FAMILY_FROZEN"
DERIVATION_FAILED = "PB4C0_FEASIBILITY_NORMALIZED_FAMILY_DERIVATION_FAILED"

PANDA_QPOS_JOINTS = [
    ("panda_joint1", "revolute", "rad"),
    ("panda_joint2", "revolute", "rad"),
    ("panda_joint3", "revolute", "rad"),
    ("panda_joint4", "revolute", "rad"),
    ("panda_joint5", "revolute", "rad"),
    ("panda_joint6", "revolute", "rad"),
    ("panda_joint7", "revolute", "rad"),
    ("panda_finger_joint1", "prismatic", "m"),
    ("panda_finger_joint2", "prismatic", "m"),
]


def _source_paths(config):
    return {
        "pb4_config": REPO_ROOT / config["source"]["pb4_config"],
        "pb4_evidence": REPO_ROOT / config["source"]["pb4_evidence"],
    }


def validate_static_rule(config):
    if config["classification"] != "CPU_ONLY_ACTION_FAMILY_DERIVATION":
        raise RuntimeError("PB4-C0 classification changed")

    rule = config["family_rule"]
    expected = {
        "intervention_space": "per_arm_joint_position_target_qpos",
        "cartesian_motion_reversal_claim": False,
        "primary_horizon_microsteps": 20,
        "arm_dof_block": 9,
        "unit_mask_template": [-1.0, 0.0, 1.0],
        "symmetric_scale_cap": 1.0,
        "feasibility_safety_factor": 0.95,
        "position_limit_tolerance_native_units": 1e-12,
        "allow_clipping": False,
        "allow_per_joint_scale": False,
        "allow_per_arm_scale": False,
        "allow_per_pair_scale": False,
        "allow_horizon_change": False,
        "uses_pb4_reward_or_regret": False,
        "uses_pb3_future_divergence": False,
        "uses_winding_labels": False,
        "require_unit_scale_infeasible": True,
        "require_formal_grid_zero_limit_violations": True,
    }
    for key, value in expected.items():
        if rule.get(key) != value:
            raise RuntimeError(
                f"PB4-C0 frozen family rule changed at {key}: "
                f"{rule.get(key)!r} != {value!r}"
            )

    if config["next_phase_boundary"]["gpu_allowed_in_this_phase"]:
        raise RuntimeError("PB4-C0 must remain CPU-only")
    if config["next_phase_boundary"]["condition5_confirmation_allowed"]:
        raise RuntimeError("PB4-C0 cannot confirm Condition 5")


def validate_sources(config):
    validate_static_rule(config)
    paths = _source_paths(config)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))

    pb4_config = load_json(paths["pb4_config"])
    pb4_evidence = load_json(paths["pb4_evidence"])

    if pb4_evidence.get("verdict") != config["source"]["expected_pb4_verdict"]:
        raise RuntimeError(
            "PB4-C0 requires the frozen complete PB4 negative result"
        )

    scientific = pb4_evidence["scientific"]
    if scientific.get("control_relevance_status") != "TESTED":
        raise RuntimeError("PB4 source was not scientifically tested")
    if not scientific["live_pair_barrier"]["valid"]:
        raise RuntimeError("PB4 source live barrier did not pass")
    if not scientific["snapshot_restore_barrier"]["valid"]:
        raise RuntimeError("PB4 source snapshot barrier did not pass")
    if int(scientific["action_evaluation_record_count"]) != 240:
        raise RuntimeError("PB4 source does not contain 240 action records")
    audit = scientific.get("audit")
    if not isinstance(audit, dict) or bool(audit.get("confirmed")):
        raise RuntimeError("PB4 source is not the frozen negative audit")

    raw, shortlist, validation, rule, pb3b2, qpos = pb4.validate_sources(
        pb4_config
    )

    dlo_gitlink = pb4.git("rev-parse", "HEAD:external/dlo-lab")
    expected_dlo = config["benchmark"]["revision"]
    if dlo_gitlink != expected_dlo:
        raise RuntimeError(
            f"DLO-Lab gitlink mismatch: {dlo_gitlink} != {expected_dlo}"
        )

    return {
        "pb4_config": pb4_config,
        "pb4_evidence": pb4_evidence,
        "raw": raw,
        "shortlist": shortlist,
        "cohort_validation": validation,
        "pb3r3_rule": rule,
        "pb3b2": pb3b2,
        "qpos": np.asarray(qpos, dtype=np.float64),
        "dlo_gitlink": dlo_gitlink,
    }


def load_pinned_panda_limits():
    urdf_path = (
        DLO_ROOT
        / "genesis"
        / "assets"
        / "urdf"
        / "panda_bullet"
        / "panda.urdf"
    )
    if not urdf_path.is_file():
        raise FileNotFoundError(str(urdf_path))

    root = ET.parse(urdf_path).getroot()
    joints = {
        node.attrib["name"]: node
        for node in root.findall("joint")
        if "name" in node.attrib
    }

    lower = []
    upper = []
    metadata = []

    for name, expected_type, unit in PANDA_QPOS_JOINTS:
        if name not in joints:
            raise RuntimeError(f"Missing Panda joint {name}")
        joint = joints[name]
        if joint.attrib.get("type") != expected_type:
            raise RuntimeError(
                f"Unexpected type for {name}: {joint.attrib.get('type')}"
            )
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"Missing hard limit for {name}")
        lo = float(limit.attrib["lower"])
        hi = float(limit.attrib["upper"])
        if not np.isfinite([lo, hi]).all() or not lo < hi:
            raise RuntimeError(f"Invalid hard limits for {name}: {lo}, {hi}")
        lower.append(lo)
        upper.append(hi)
        metadata.append(
            {
                "joint_name": name,
                "joint_type": expected_type,
                "unit": unit,
                "lower": lo,
                "upper": hi,
            }
        )

    return {
        "urdf_path": str(urdf_path),
        "joint_names": [row[0] for row in PANDA_QPOS_JOINTS],
        "lower": np.asarray(lower, dtype=np.float64),
        "upper": np.asarray(upper, dtype=np.float64),
        "metadata": metadata,
    }


def symmetric_alpha_limit(base, delta, lower, upper):
    """Largest symmetric |mask| for q=base+mask*delta within [lower,upper]."""
    base = float(base)
    delta = float(delta)
    lower = float(lower)
    upper = float(upper)

    if not np.isfinite([base, delta, lower, upper]).all():
        raise ValueError("Non-finite alpha-limit input")
    if lower >= upper:
        raise ValueError("Invalid position limits")
    if base < lower or base > upper:
        raise ValueError(
            f"Base qpos outside hard limit: q={base}, [{lower},{upper}]"
        )
    if delta == 0.0:
        return float("inf")

    lower_slack = base - lower
    upper_slack = upper - base
    return float(min(lower_slack, upper_slack) / abs(delta))


def limiting_signed_mask(delta, lower_slack, upper_slack):
    """Return which unit sign reaches the limiting bound."""
    delta = float(delta)
    if delta == 0.0:
        return None
    if lower_slack <= upper_slack:
        # To move toward lower: sign(mask)*sign(delta) < 0.
        return -1 if delta > 0 else 1
    # To move toward upper: sign(mask)*sign(delta) > 0.
    return 1 if delta > 0 else -1


def branch_times(shortlist):
    times = sorted({int(pair["time_index"]) for pair in shortlist["pairs"]})
    if not times:
        raise RuntimeError("Frozen cohort has no branch times")
    return times


def command_for_mask_pair(qpos, time_index, relative_step, mask_a, mask_b):
    qpos = np.asarray(qpos, dtype=np.float64)
    t = int(time_index)
    r = int(relative_step)
    base = qpos[t]
    nominal = qpos[t + r]
    out = base.copy()
    out[:9] = base[:9] + float(mask_a) * (nominal[:9] - base[:9])
    out[9:] = base[9:] + float(mask_b) * (nominal[9:] - base[9:])
    if out.shape != (18,) or not np.isfinite(out).all():
        raise RuntimeError("Generated non-finite/invalid 18D qpos target")
    return out


def command_limit_violations(command, limits, tolerance):
    command = np.asarray(command, dtype=np.float64)
    violations = []
    for arm_index, block in enumerate((command[:9], command[9:]), start=1):
        below = block < (limits["lower"] - float(tolerance))
        above = block > (limits["upper"] + float(tolerance))
        for joint_index in np.flatnonzero(below | above):
            joint_index = int(joint_index)
            violations.append(
                {
                    "arm_index": arm_index,
                    "joint_index": joint_index,
                    "joint_name": limits["joint_names"][joint_index],
                    "value": float(block[joint_index]),
                    "lower": float(limits["lower"][joint_index]),
                    "upper": float(limits["upper"][joint_index]),
                }
            )
    return violations


def count_grid_violations(
        qpos,
        times,
        horizon,
        mask_values,
        limits,
        tolerance):
    command_count = 0
    violating_command_count = 0
    violation_event_count = 0
    first_violation = None

    for t in times:
        for r in range(1, int(horizon) + 1):
            for mask_a in mask_values:
                for mask_b in mask_values:
                    command = command_for_mask_pair(
                        qpos,
                        t,
                        r,
                        mask_a,
                        mask_b,
                    )
                    command_count += 1
                    violations = command_limit_violations(
                        command,
                        limits,
                        tolerance,
                    )
                    if violations:
                        violating_command_count += 1
                        violation_event_count += len(violations)
                        if first_violation is None:
                            first_violation = {
                                "time_index": int(t),
                                "relative_step": int(r),
                                "arm1_mask": float(mask_a),
                                "arm2_mask": float(mask_b),
                                "violations": violations,
                            }

    return {
        "generated_command_count": int(command_count),
        "out_of_limit_command_count": int(violating_command_count),
        "joint_limit_violation_event_count": int(violation_event_count),
        "first_violation": first_violation,
    }


def derive_alpha_max(qpos, times, horizon, limits):
    qpos = np.asarray(qpos, dtype=np.float64)
    alpha_max = float("inf")
    limiting = None
    constraint_count = 0

    for t in times:
        if int(t) + int(horizon) >= len(qpos):
            raise RuntimeError(
                f"Horizon exceeds qpos at t={t}, H={horizon}, T={len(qpos)}"
            )
        base_full = qpos[int(t)]

        for r in range(1, int(horizon) + 1):
            nominal_full = qpos[int(t) + r]
            for arm_index, (start, stop) in enumerate(
                    ((0, 9), (9, 18)),
                    start=1):
                base = base_full[start:stop]
                delta = nominal_full[start:stop] - base

                for joint_index in range(9):
                    b = float(base[joint_index])
                    d = float(delta[joint_index])
                    lo = float(limits["lower"][joint_index])
                    hi = float(limits["upper"][joint_index])

                    if b < lo or b > hi:
                        raise RuntimeError(
                            "Frozen branch-time qpos itself violates pinned "
                            f"hard limit: t={t}, arm={arm_index}, "
                            f"joint={limits['joint_names'][joint_index]}, "
                            f"q={b}, limits=[{lo},{hi}]"
                        )
                    if d == 0.0:
                        continue

                    constraint_count += 1
                    lower_slack = b - lo
                    upper_slack = hi - b
                    alpha_limit = symmetric_alpha_limit(
                        b,
                        d,
                        lo,
                        hi,
                    )
                    if alpha_limit < alpha_max:
                        alpha_max = float(alpha_limit)
                        side = (
                            "lower"
                            if lower_slack <= upper_slack
                            else "upper"
                        )
                        limiting = {
                            "time_index": int(t),
                            "relative_step": int(r),
                            "arm_index": int(arm_index),
                            "joint_index": int(joint_index),
                            "joint_name": limits["joint_names"][joint_index],
                            "joint_type":
                                limits["metadata"][joint_index]["joint_type"],
                            "unit": limits["metadata"][joint_index]["unit"],
                            "base_qpos": b,
                            "nominal_qpos": float(
                                nominal_full[start + joint_index]
                            ),
                            "nominal_delta": d,
                            "lower": lo,
                            "upper": hi,
                            "lower_slack": float(lower_slack),
                            "upper_slack": float(upper_slack),
                            "limiting_bound": side,
                            "limiting_unit_mask_sign": int(
                                limiting_signed_mask(
                                    d,
                                    lower_slack,
                                    upper_slack,
                                )
                            ),
                            "alpha_limit": float(alpha_limit),
                        }

    if constraint_count == 0 or not np.isfinite(alpha_max):
        raise RuntimeError("No finite nonzero qpos-delta constraint found")
    if alpha_max <= 0.0:
        raise RuntimeError(f"Non-positive alpha_max: {alpha_max}")

    return {
        "alpha_max_hard_limits": float(alpha_max),
        "constraint_count": int(constraint_count),
        "limiting_constraint": limiting,
    }


def min_limit_margins(
        qpos,
        times,
        horizon,
        mask_values,
        limits):
    min_revolute = float("inf")
    min_prismatic = float("inf")

    for t in times:
        for r in range(1, int(horizon) + 1):
            for mask_a in mask_values:
                for mask_b in mask_values:
                    command = command_for_mask_pair(
                        qpos,
                        t,
                        r,
                        mask_a,
                        mask_b,
                    )
                    for block in (command[:9], command[9:]):
                        margin = np.minimum(
                            block - limits["lower"],
                            limits["upper"] - block,
                        )
                        min_revolute = min(
                            min_revolute,
                            float(np.min(margin[:7])),
                        )
                        min_prismatic = min(
                            min_prismatic,
                            float(np.min(margin[7:])),
                        )

    return {
        "minimum_revolute_joint_limit_margin_rad":
            float(min_revolute),
        "minimum_prismatic_finger_limit_margin_m":
            float(min_prismatic),
    }


def exact_action_grid(alpha):
    alpha = float(alpha)
    return [
        {
            "action_id": "both_negative",
            "arm1_mask": -alpha,
            "arm2_mask": -alpha,
        },
        {
            "action_id": "arm1_negative_arm2_hold",
            "arm1_mask": -alpha,
            "arm2_mask": 0.0,
        },
        {
            "action_id": "arm1_negative_arm2_positive",
            "arm1_mask": -alpha,
            "arm2_mask": alpha,
        },
        {
            "action_id": "arm1_hold_arm2_negative",
            "arm1_mask": 0.0,
            "arm2_mask": -alpha,
        },
        {
            "action_id": "hold_both",
            "arm1_mask": 0.0,
            "arm2_mask": 0.0,
        },
        {
            "action_id": "arm1_hold_arm2_positive",
            "arm1_mask": 0.0,
            "arm2_mask": alpha,
        },
        {
            "action_id": "arm1_positive_arm2_negative",
            "arm1_mask": alpha,
            "arm2_mask": -alpha,
        },
        {
            "action_id": "arm1_positive_arm2_hold",
            "arm1_mask": alpha,
            "arm2_mask": 0.0,
        },
        {
            "action_id": "both_positive",
            "arm1_mask": alpha,
            "arm2_mask": alpha,
        },
    ]


def build_rule(config, sources):
    return {
        "phase": "PB4-C0",
        "verdict": RULE_VERDICT,
        "classification": config["classification"],
        "source": {
            "pb4_verdict": sources["pb4_evidence"]["verdict"],
            "cohort_pair_count":
                int(sources["cohort_validation"]["pair_count"]),
            "unique_rollout_count":
                int(sources["cohort_validation"]["unique_rollout_count"]),
            "dlo_gitlink": sources["dlo_gitlink"],
        },
        "family_rule": config["family_rule"],
        "derivation_inputs": {
            "frozen_best_qpos_only": True,
            "frozen_cohort_branch_times_only": True,
            "pinned_hard_joint_limits_only": True,
            "pb4_reward_or_regret_used": False,
            "pb3_future_divergence_used": False,
            "winding_labels_used": False,
        },
        "phase_boundary": config["next_phase_boundary"],
    }


def freeze_rule(config_path):
    config = load_json(config_path)
    sources = validate_sources(config)
    rule = build_rule(config, sources)

    path = REPO_ROOT / config["outputs"]["rule"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rule, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"verdict={RULE_VERDICT}")
    print(f"rule={path}")
    print("gpu_allowed=false")
    return rule


def load_and_validate_rule(config, sources):
    path = REPO_ROOT / config["outputs"]["rule"]
    if not path.is_file():
        raise FileNotFoundError(str(path))
    actual = load_json(path)
    expected = build_rule(config, sources)
    if actual != expected:
        raise RuntimeError(
            "Committed PB4-C0 rule differs from deterministic preregistration"
        )
    return actual


def derive_family_payload(config, sources, rule):
    family_rule = config["family_rule"]
    qpos = sources["qpos"]
    shortlist = sources["shortlist"]
    times = branch_times(shortlist)
    horizon = int(family_rule["primary_horizon_microsteps"])
    tolerance = float(
        family_rule["position_limit_tolerance_native_units"]
    )
    limits = load_pinned_panda_limits()

    unit_grid = count_grid_violations(
        qpos,
        times,
        horizon,
        [-1.0, 0.0, 1.0],
        limits,
        tolerance,
    )
    if (
        family_rule["require_unit_scale_infeasible"]
        and unit_grid["out_of_limit_command_count"] == 0
    ):
        raise RuntimeError(
            "Unit signed grid unexpectedly has zero hard-limit violations; "
            "PB4-C0 normalization rationale is not reproduced."
        )

    alpha = derive_alpha_max(
        qpos,
        times,
        horizon,
        limits,
    )
    alpha_cap = float(family_rule["symmetric_scale_cap"])
    safety = float(family_rule["feasibility_safety_factor"])
    alpha_reference = min(
        float(alpha["alpha_max_hard_limits"]),
        alpha_cap,
    )
    alpha_formal = float(safety * alpha_reference)

    if not (0.0 < alpha_formal <= alpha_cap):
        raise RuntimeError(f"Invalid derived alpha_formal: {alpha_formal}")

    mask_values = [-alpha_formal, 0.0, alpha_formal]
    formal_grid = count_grid_violations(
        qpos,
        times,
        horizon,
        mask_values,
        limits,
        tolerance,
    )
    if (
        family_rule["require_formal_grid_zero_limit_violations"]
        and formal_grid["out_of_limit_command_count"] != 0
    ):
        raise RuntimeError(
            "Derived formal signed qpos-target grid still violates hard limits"
        )

    margins = min_limit_margins(
        qpos,
        times,
        horizon,
        mask_values,
        limits,
    )

    return {
        "phase": "PB4-C0",
        "verdict": SUCCESS_VERDICT,
        "classification": "CPU_ONLY_ACTION_FAMILY_DERIVATION",
        "formal_condition5_confirmed": False,
        "gpu_executed": False,
        "source_rule_verdict": rule["verdict"],
        "intervention_space": "per_arm_joint_position_target_qpos",
        "cartesian_motion_reversal_claim": False,
        "branch_times": times,
        "primary_horizon_microsteps": horizon,
        "position_limits": {
            "source": family_rule["position_limit_source"],
            "urdf_path": limits["urdf_path"],
            "joint_metadata": limits["metadata"],
            "comparison_tolerance_native_units": tolerance,
        },
        "unit_scale_diagnostic": unit_grid,
        "derivation": {
            **alpha,
            "symmetric_scale_cap": alpha_cap,
            "feasibility_safety_factor": safety,
            "alpha_reference_before_safety_factor":
                float(alpha_reference),
            "alpha_formal": alpha_formal,
            "formula": family_rule["formal_alpha_definition"],
            "uses_reward_or_regret": False,
            "uses_future_divergence": False,
            "uses_winding_labels": False,
        },
        "formal_family": {
            "mask_values": mask_values,
            "actions": exact_action_grid(alpha_formal),
            "allow_clipping": False,
            "allow_per_joint_scale": False,
            "allow_per_arm_scale": False,
            "allow_per_pair_scale": False,
            "allow_horizon_change": False,
            "joint_limit_validation": (
                formal_grid["out_of_limit_command_count"] == 0
            ),
            "out_of_limit_command_count":
                int(formal_grid["out_of_limit_command_count"]),
            "generated_command_count":
                int(formal_grid["generated_command_count"]),
            **margins,
        },
        "next_phase_boundary": {
            "gpu_allowed_in_pb4c0": False,
            "pb4c1_automatically_started": False,
            "next_if_success": (
                "STOP. A separate PB4-C1 exploratory GPU protocol may use "
                "this exact frozen alpha_formal and action family."
            ),
        },
    }


def write_result(config, sources, rule, family):
    family_path = REPO_ROOT / config["outputs"]["family"]
    family_path.parent.mkdir(parents=True, exist_ok=True)
    family_path.write_text(
        json.dumps(family, indent=2) + "\n",
        encoding="utf-8",
    )

    report_dir = REPO_ROOT / config["outputs"]["committed_report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    evidence = {
        "phase_name": config["phase_name"],
        "verdict": family["verdict"],
        "repository": {
            "starting_main_sha":
                config["provenance"]["starting_main_sha"],
            "ending_main_sha": pb4.git("rev-parse", "HEAD"),
            "dlolab_gitlink": sources["dlo_gitlink"],
        },
        "scientific": {
            "classification": family["classification"],
            "formal_condition5_confirmed": False,
            "gpu_executed": False,
            "rule": rule,
            "family": family,
            "boundaries": {
                "pb4_result_modified": False,
                "gpu_started": False,
                "sensor_started": False,
                "training_started": False,
            },
        },
    }
    (report_dir / "EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    limiting = family["derivation"]["limiting_constraint"]
    lines = [
        "# PB4-C0 Feasibility-Normalized Signed Qpos-Target Family",
        "",
        f"Verdict: `{family['verdict']}`",
        "",
        "**Classification:** `CPU_ONLY_ACTION_FAMILY_DERIVATION`",
        "",
        "**Formal Condition-5 confirmed:** `false`",
        "",
        "**GPU executed:** `false`",
        "",
        "## Unit-scale diagnostic",
        "",
        (
            "- Unit {-1,0,+1} grid violating commands: "
            f"{family['unit_scale_diagnostic']['out_of_limit_command_count']}"
        ),
        (
            "- Unit grid generated commands: "
            f"{family['unit_scale_diagnostic']['generated_command_count']}"
        ),
        "",
        "## Derived global scale",
        "",
        (
            "- alpha_max_hard_limits: "
            f"{family['derivation']['alpha_max_hard_limits']:.17g}"
        ),
        (
            "- alpha_reference_before_safety_factor: "
            f"{family['derivation']['alpha_reference_before_safety_factor']:.17g}"
        ),
        (
            "- feasibility_safety_factor: "
            f"{family['derivation']['feasibility_safety_factor']:.17g}"
        ),
        (
            "- alpha_formal: "
            f"{family['derivation']['alpha_formal']:.17g}"
        ),
        "",
        "## Limiting constraint",
        "",
        f"- time: {limiting['time_index']}",
        f"- relative step: {limiting['relative_step']}",
        f"- arm: {limiting['arm_index']}",
        f"- joint: {limiting['joint_name']}",
        f"- limiting bound: {limiting['limiting_bound']}",
        f"- limiting unit-mask sign: {limiting['limiting_unit_mask_sign']}",
        "",
        "## Formal grid validation",
        "",
        (
            "- out-of-limit commands: "
            f"{family['formal_family']['out_of_limit_command_count']}"
        ),
        (
            "- minimum revolute limit margin: "
            f"{family['formal_family']['minimum_revolute_joint_limit_margin_rad']:.17g} rad"
        ),
        (
            "- minimum finger limit margin: "
            f"{family['formal_family']['minimum_prismatic_finger_limit_margin_m']:.17g} m"
        ),
        "",
        "## Boundary",
        "",
        (
            "PB4-C0 derives and freezes an engineering-feasible signed qpos-target "
            "family only. It does not test control relevance and does not authorize "
            "Condition-5 claims."
        ),
        "",
        (
            "STOP after committing this result. PB4-C1, if pursued, must be a "
            "separate preregistered exploratory GPU phase using the exact frozen "
            "alpha_formal."
        ),
        "",
    ]
    (report_dir / "RESULT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return evidence


def derive_family(config_path):
    config = load_json(config_path)
    sources = validate_sources(config)
    rule = load_and_validate_rule(config, sources)

    try:
        family = derive_family_payload(
            config,
            sources,
            rule,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{DERIVATION_FAILED}: {exc}"
        ) from exc

    write_result(
        config,
        sources,
        rule,
        family,
    )

    print(f"verdict={family['verdict']}")
    print(
        "unit_scale_out_of_limit_command_count="
        f"{family['unit_scale_diagnostic']['out_of_limit_command_count']}"
    )
    print(
        "alpha_max_hard_limits="
        f"{family['derivation']['alpha_max_hard_limits']:.17g}"
    )
    print(
        "alpha_formal="
        f"{family['derivation']['alpha_formal']:.17g}"
    )
    print(
        "formal_grid_out_of_limit_command_count="
        f"{family['formal_family']['out_of_limit_command_count']}"
    )
    print("gpu_executed=false")
    return family


def validate_result(config_path):
    config = load_json(config_path)
    sources = validate_sources(config)
    rule = load_and_validate_rule(config, sources)

    family_path = REPO_ROOT / config["outputs"]["family"]
    evidence_path = (
        REPO_ROOT
        / config["outputs"]["committed_report_dir"]
        / "EVIDENCE.json"
    )
    for path in (family_path, evidence_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    actual_family = load_json(family_path)
    expected_family = derive_family_payload(
        config,
        sources,
        rule,
    )
    if actual_family != expected_family:
        raise RuntimeError(
            "PB4-C0 frozen family differs from deterministic recomputation"
        )

    evidence = load_json(evidence_path)
    if evidence.get("verdict") != SUCCESS_VERDICT:
        raise RuntimeError("PB4-C0 evidence verdict mismatch")
    scientific = evidence["scientific"]
    if scientific.get("formal_condition5_confirmed") is not False:
        raise RuntimeError("PB4-C0 must keep Condition 5 unconfirmed")
    if scientific.get("gpu_executed") is not False:
        raise RuntimeError("PB4-C0 must remain CPU-only")
    if scientific["family"] != expected_family:
        raise RuntimeError("PB4-C0 evidence family mismatch")

    print("PB4-C0 result validation: PASS")
    print(f"verdict={SUCCESS_VERDICT}")
    print(
        "alpha_formal="
        f"{expected_family['derivation']['alpha_formal']:.17g}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "command",
        choices=("freeze-rule", "derive-family", "validate-result"),
    )
    args = parser.parse_args()

    if args.command == "freeze-rule":
        freeze_rule(args.config)
    elif args.command == "derive-family":
        derive_family(args.config)
    else:
        validate_result(args.config)


if __name__ == "__main__":
    main()
