"""PB4-C1 — exploratory control audit with the frozen PB4-C0 qpos family.

PB4-C1 is an exploratory GPU audit only. It reuses the already-observed PB3-B1
cohort, so even a positive result cannot confirm Condition 5.

The action family is not derived here. It must be loaded exactly from:
  configs/experiment3/published_benchmark/dlolab_wrapping_pb4c_family.json

PB4-C1 must not recompute, round, rescale, or clip alpha; change the nine action
IDs/order; change H=20, 3 repeats, 0.05, 5x, or 3-pair/2-stratum rules; change
benchmark physics/reward/geometry; or start sensing/model training.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from scripts.experiment3.dlolab_wrapping.paths import REPO_ROOT
from scripts.experiment3.dlolab_wrapping.replay_winding_audit import load_json
import scripts.experiment3.dlolab_wrapping.control_relevance_pb4 as pb4
import scripts.experiment3.dlolab_wrapping.feasibility_normalized_qpos_family_pb4c0 as pb4c0


PROTOCOL_VERDICT = "PB4C1_NORMALIZED_QPOS_PROTOCOL_PREREGISTERED"
SIGNAL_FOUND = "PB4C1_NORMALIZED_QPOS_SIGNAL_FOUND"
SIGNAL_NOT_FOUND = "PB4C1_NORMALIZED_QPOS_SIGNAL_NOT_FOUND"
LIVE_FAILED = "PB4C1_LIVE_COHORT_REVALIDATION_FAILED"
SNAPSHOT_FAILED = "PB4C1_SNAPSHOT_RESTORE_FAILED"
ACTION_FAILED = "PB4C1_ACTION_EVALUATION_FAILED"

PB4_BLOCKED_MAP = {
    pb4.LIVE_FAILED: LIVE_FAILED,
    pb4.SNAPSHOT_FAILED: SNAPSHOT_FAILED,
    pb4.ACTION_FAILED: ACTION_FAILED,
}

EXPECTED_ACTION_IDS = [
    "both_negative",
    "arm1_negative_arm2_hold",
    "arm1_negative_arm2_positive",
    "arm1_hold_arm2_negative",
    "hold_both",
    "arm1_hold_arm2_positive",
    "arm1_positive_arm2_negative",
    "arm1_positive_arm2_hold",
    "both_positive",
]


def validate_static_config(config):
    if config["classification"] != "EXPLORATORY":
        raise RuntimeError("PB4-C1 classification must remain EXPLORATORY")

    family_lock = config["family_lock"]
    expected_false = (
        "recompute_alpha",
        "round_alpha",
        "rescale_alpha",
        "clip_qpos_targets",
        "modify_action_ids",
        "modify_action_order",
        "modify_horizon",
    )
    for key in expected_false:
        if family_lock.get(key) is not False:
            raise RuntimeError(f"PB4-C1 family lock changed: {key}")
    if not family_lock.get("require_zero_hard_limit_violations_pre_gpu"):
        raise RuntimeError("PB4-C1 must revalidate zero hard-limit violations")

    rules = config["control_rules"]
    expected_rules = {
        "inherit_from_pb4": True,
        "primary_horizon_microsteps": 20,
        "repeat_count": 3,
        "absolute_regret_min": 0.05,
        "repeat_floor_multiplier": 5.0,
        "minimum_passing_pairs": 3,
        "minimum_passing_winding_strata": 2,
        "winding_strata_source":
            "PB4-C1 fresh live revalidation winding indices at branch time",
    }
    for key, value in expected_rules.items():
        if rules.get(key) != value:
            raise RuntimeError(
                f"PB4-C1 frozen control rule changed at {key}: "
                f"{rules.get(key)!r} != {value!r}"
            )

    scope = config["exploratory_scope"]
    if not scope["same_cohort_reused_after_pb4_outcome_seen"]:
        raise RuntimeError("PB4-C1 must explicitly acknowledge cohort reuse")
    if scope["formal_condition5_confirmed"] is not False:
        raise RuntimeError("PB4-C1 cannot mark Condition 5 confirmed")
    if scope["condition5_confirmation_allowed"] is not False:
        raise RuntimeError("PB4-C1 cannot allow Condition-5 confirmation")
    if scope["positive_verdict"] != SIGNAL_FOUND:
        raise RuntimeError("PB4-C1 positive verdict changed")
    if scope["negative_verdict"] != SIGNAL_NOT_FOUND:
        raise RuntimeError("PB4-C1 negative verdict changed")


def source_paths(config):
    source = config["source"]
    return {
        "pb4_config": REPO_ROOT / source["pb4_config"],
        "pb4_evidence": REPO_ROOT / source["pb4_evidence"],
        "pb4c0_family": REPO_ROOT / source["pb4c0_family"],
        "pb4c0_evidence": REPO_ROOT / source["pb4c0_evidence"],
    }


def validate_frozen_family(family, c0_evidence, shortlist, qpos):
    if family.get("verdict") != "PB4C0_FEASIBILITY_NORMALIZED_FAMILY_FROZEN":
        raise RuntimeError("PB4-C1 family verdict mismatch")
    if family.get("classification") != "CPU_ONLY_ACTION_FAMILY_DERIVATION":
        raise RuntimeError("PB4-C1 source family classification mismatch")
    if family.get("formal_condition5_confirmed") is not False:
        raise RuntimeError("PB4-C0 family must keep Condition 5 false")
    if family.get("gpu_executed") is not False:
        raise RuntimeError("PB4-C0 family must be CPU-only")
    if family.get("intervention_space") != "per_arm_joint_position_target_qpos":
        raise RuntimeError("PB4-C0 family intervention space changed")
    if family.get("cartesian_motion_reversal_claim") is not False:
        raise RuntimeError("PB4-C0 family must not claim Cartesian reversal")
    if int(family["primary_horizon_microsteps"]) != 20:
        raise RuntimeError("PB4-C0 family H changed")

    evidence_family = c0_evidence["scientific"]["family"]
    if evidence_family != family:
        raise RuntimeError(
            "PB4-C0 family file differs from committed PB4-C0 evidence"
        )
    if c0_evidence["scientific"].get("formal_condition5_confirmed") is not False:
        raise RuntimeError("PB4-C0 evidence must keep Condition 5 false")
    if c0_evidence["scientific"].get("gpu_executed") is not False:
        raise RuntimeError("PB4-C0 evidence unexpectedly says GPU executed")

    alpha = float(family["derivation"]["alpha_formal"])
    if not np.isfinite(alpha) or not (0.0 < alpha <= 1.0):
        raise RuntimeError(f"Invalid frozen alpha_formal: {alpha}")

    formal = family["formal_family"]
    if formal.get("joint_limit_validation") is not True:
        raise RuntimeError("PB4-C0 family did not pass hard-limit validation")
    if int(formal.get("out_of_limit_command_count", -1)) != 0:
        raise RuntimeError("PB4-C0 family contains hard-limit violations")
    for key in (
        "allow_clipping",
        "allow_per_joint_scale",
        "allow_per_arm_scale",
        "allow_per_pair_scale",
        "allow_horizon_change",
    ):
        if formal.get(key) is not False:
            raise RuntimeError(f"PB4-C0 family mutation flag enabled: {key}")

    mask_values = [float(v) for v in formal["mask_values"]]
    expected_masks = [-alpha, 0.0, alpha]
    if mask_values != expected_masks:
        raise RuntimeError(
            f"PB4-C0 mask values changed: {mask_values} != {expected_masks}"
        )

    actions = formal["actions"]
    if [row["action_id"] for row in actions] != EXPECTED_ACTION_IDS:
        raise RuntimeError("PB4-C0 action IDs/order changed")

    expected_mask_pairs = [
        (-alpha, -alpha),
        (-alpha, 0.0),
        (-alpha, alpha),
        (0.0, -alpha),
        (0.0, 0.0),
        (0.0, alpha),
        (alpha, -alpha),
        (alpha, 0.0),
        (alpha, alpha),
    ]
    actual_mask_pairs = [
        (float(row["arm1_mask"]), float(row["arm2_mask"]))
        for row in actions
    ]
    if actual_mask_pairs != expected_mask_pairs:
        raise RuntimeError("PB4-C0 action masks/order changed")

    times = pb4c0.branch_times(shortlist)
    if [int(v) for v in family["branch_times"]] != times:
        raise RuntimeError(
            "PB4-C0 frozen branch times differ from current frozen cohort"
        )

    limits = pb4c0.load_pinned_panda_limits()
    tolerance = float(
        family["position_limits"]["comparison_tolerance_native_units"]
    )
    preflight = pb4c0.count_grid_violations(
        qpos,
        times,
        20,
        mask_values,
        limits,
        tolerance,
    )
    if int(preflight["out_of_limit_command_count"]) != 0:
        raise RuntimeError(
            "PB4-C1 pre-GPU revalidation found a hard-limit violation in the "
            "frozen PB4-C0 family. Do not run GPU."
        )
    margins = pb4c0.min_limit_margins(
        qpos,
        times,
        20,
        mask_values,
        limits,
    )

    return {
        "alpha_formal": alpha,
        "mask_values": mask_values,
        "actions": copy.deepcopy(actions),
        "branch_times": times,
        "pre_gpu_hard_limit_validation": {
            **preflight,
            **margins,
            "joint_limit_validation":
                int(preflight["out_of_limit_command_count"]) == 0,
            "position_limit_source": family["position_limits"]["source"],
            "comparison_tolerance_native_units": tolerance,
        },
    }


def validate_sources(config):
    validate_static_config(config)
    paths = source_paths(config)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))

    pb4_config = load_json(paths["pb4_config"])
    pb4_evidence = load_json(paths["pb4_evidence"])
    family = load_json(paths["pb4c0_family"])
    c0_evidence = load_json(paths["pb4c0_evidence"])

    if pb4_evidence.get("verdict") != config["source"]["expected_pb4_verdict"]:
        raise RuntimeError("PB4-C1 requires the frozen PB4 negative")
    pb4_scientific = pb4_evidence["scientific"]
    if pb4_scientific.get("control_relevance_status") != "TESTED":
        raise RuntimeError("PB4 source is not a completed scientific test")
    if not pb4_scientific["live_pair_barrier"]["valid"]:
        raise RuntimeError("PB4 source live barrier did not pass")
    if not pb4_scientific["snapshot_restore_barrier"]["valid"]:
        raise RuntimeError("PB4 source snapshot barrier did not pass")
    if int(pb4_scientific["action_evaluation_record_count"]) != 240:
        raise RuntimeError("PB4 source does not contain 240 action records")
    pb4_audit = pb4_scientific.get("audit")
    if not isinstance(pb4_audit, dict) or bool(pb4_audit.get("confirmed")):
        raise RuntimeError("PB4 source must be the frozen negative audit")

    if c0_evidence.get("verdict") != config["source"]["expected_pb4c0_verdict"]:
        raise RuntimeError("PB4-C1 requires the frozen PB4-C0 family result")

    raw, shortlist, validation, rule, pb3b2, qpos = pb4.validate_sources(
        pb4_config
    )

    dlo_gitlink = pb4.git("rev-parse", "HEAD:external/dlo-lab")
    if dlo_gitlink != config["benchmark"]["revision"]:
        raise RuntimeError(
            f"DLO-Lab gitlink mismatch: {dlo_gitlink} != "
            f"{config['benchmark']['revision']}"
        )

    family_validation = validate_frozen_family(
        family,
        c0_evidence,
        shortlist,
        qpos,
    )

    return {
        "pb4_config": pb4_config,
        "pb4_evidence": pb4_evidence,
        "pb4c0_family": family,
        "pb4c0_evidence": c0_evidence,
        "raw": raw,
        "shortlist": shortlist,
        "cohort_validation": validation,
        "pb3r3_rule": rule,
        "pb3b2": pb3b2,
        "qpos": np.asarray(qpos, dtype=np.float64),
        "dlo_gitlink": dlo_gitlink,
        "family_validation": family_validation,
    }


def build_runtime_config(config, sources):
    """Create the exact PB4 execution config without recomputing alpha."""
    runtime = copy.deepcopy(sources["pb4_config"])
    frozen = sources["family_validation"]

    runtime["phase_name"] = config["phase_name"]
    runtime["action_library"] = {
        "interface": "verified_best_qpos_position_control_suffix",
        "primary_horizon_microsteps": 20,
        "macro_step_equivalents": 2,
        "arm_dof_block": 9,
        "construction": (
            "For branch time t and relative microstep r, each 9-DOF arm qpos "
            "block is q_t_arm + mask_arm * (q_nominal[t+r]_arm - q_t_arm), "
            "using the exact frozen PB4-C0 mask values and action order."
        ),
        "intervention_space": "per_arm_joint_position_target_qpos",
        "cartesian_motion_reversal_claim": False,
        "alpha_source": config["family_lock"]["source_of_truth"],
        "alpha_formal": frozen["alpha_formal"],
        "mask_values": frozen["mask_values"],
        "actions": copy.deepcopy(frozen["actions"]),
        "recompute_alpha": False,
        "rescale_alpha": False,
        "round_alpha": False,
        "allow_clipping": False,
        "allow_action_addition_after_gpu": False,
        "allow_action_removal_after_gpu": False,
    }

    runtime["control_relevance"] = copy.deepcopy(
        sources["pb4_config"]["control_relevance"]
    )
    runtime["control_relevance"]["repeat_count"] = 3
    runtime["control_relevance"]["absolute_regret_min"] = 0.05
    runtime["control_relevance"]["repeat_floor_multiplier"] = 5.0
    runtime["control_relevance"]["minimum_passing_pairs"] = 3
    runtime["control_relevance"]["minimum_passing_winding_strata"] = 2
    runtime["control_relevance"]["winding_strata_source"] = (
        config["control_rules"]["winding_strata_source"]
    )
    runtime["control_relevance"]["historical_pb3b1_stratum_is_formal_gate"] = False
    runtime["control_relevance"]["positive_verdict"] = SIGNAL_FOUND
    runtime["control_relevance"]["negative_verdict"] = SIGNAL_NOT_FOUND

    runtime["outputs"] = copy.deepcopy(config["outputs"])
    runtime["provenance"] = copy.deepcopy(config["provenance"])
    return runtime


def build_protocol(config, sources, runtime):
    fv = sources["family_validation"]
    cr = runtime["control_relevance"]
    return {
        "phase": "PB4-C1",
        "verdict": PROTOCOL_VERDICT,
        "classification": "EXPLORATORY",
        "formal_condition5_confirmed": False,
        "condition5_confirmation_allowed": False,
        "scientific_question": (
            "Does the exact feasibility-normalized signed qpos-target family "
            "frozen by PB4-C0 show repeat-robust branch-dependent control "
            "preference on the already-observed cohort?"
        ),
        "source": {
            "starting_pb4_verdict": sources["pb4_evidence"]["verdict"],
            "pb4c0_verdict": sources["pb4c0_family"]["verdict"],
            "same_cohort_reused_after_pb4_outcome_seen": True,
            "cohort_pair_count": int(sources["cohort_validation"]["pair_count"]),
            "unique_rollout_count": int(sources["cohort_validation"]["unique_rollout_count"]),
            "dlo_gitlink": sources["dlo_gitlink"],
        },
        "frozen_family": {
            "source_of_truth": config["family_lock"]["source_of_truth"],
            "alpha_formal": fv["alpha_formal"],
            "mask_values": fv["mask_values"],
            "actions": fv["actions"],
            "primary_horizon_microsteps": 20,
            "recompute_alpha": False,
            "round_alpha": False,
            "rescale_alpha": False,
            "clip_qpos_targets": False,
            "modify_action_ids": False,
            "modify_action_order": False,
            "pre_gpu_hard_limit_validation": fv["pre_gpu_hard_limit_validation"],
        },
        "task_score": copy.deepcopy(runtime["task_score"]),
        "barriers": {
            "all_10_live_pairs_before_snapshot": True,
            "live_prefix_validity_required": True,
            "live_prefix_validity_rule": runtime["barriers"]["live_prefix_validity_rule"],
            "all_20_branches_restore_before_actions": True,
            "snapshot_restore_rope_max_abs_m": float(runtime["barriers"]["snapshot_restore_rope_max_abs_m"]),
            "pair_drop_allowed": False,
            "pair_replacement_allowed": False,
        },
        "control_relevance": {
            "repeat_count": int(cr["repeat_count"]),
            "best_action_statistic": cr["best_action_statistic"],
            "repeat_floor": cr["repeat_floor"],
            "absolute_regret_min": float(cr["absolute_regret_min"]),
            "repeat_floor_multiplier": float(cr["repeat_floor_multiplier"]),
            "unique_best_rule": cr["unique_best_rule"],
            "robust_cross_regret": cr["robust_cross_regret"],
            "pair_pass_rule": cr["pair_pass_rule"],
            "minimum_passing_pairs": int(cr["minimum_passing_pairs"]),
            "minimum_passing_winding_strata": int(cr["minimum_passing_winding_strata"]),
            "winding_strata_source": config["control_rules"]["winding_strata_source"],
        },
        "expected_action_evaluation_records": 540,
        "stop_rule": copy.deepcopy(config["exploratory_scope"]),
        "phase_boundary": {
            "deployable_sensor_test_started": False,
            "state_diff_training_started": False,
            "cfpm_training_started": False,
        },
    }


def freeze_protocol(config_path):
    config = load_json(config_path)
    sources = validate_sources(config)
    runtime = build_runtime_config(config, sources)
    protocol = build_protocol(config, sources, runtime)

    path = REPO_ROOT / config["outputs"]["protocol"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    print(f"verdict={PROTOCOL_VERDICT}")
    print(f"protocol={path}")
    print(f"alpha_formal={protocol['frozen_family']['alpha_formal']:.17g}")
    print("action_count=9")
    print("expected_action_records=540")
    print(
        "pre_gpu_out_of_limit_command_count="
        f"{protocol['frozen_family']['pre_gpu_hard_limit_validation']['out_of_limit_command_count']}"
    )
    return protocol


def load_and_validate_protocol(config, sources, runtime):
    path = REPO_ROOT / config["outputs"]["protocol"]
    if not path.is_file():
        raise FileNotFoundError(str(path))
    actual = load_json(path)
    expected = build_protocol(config, sources, runtime)
    if actual != expected:
        raise RuntimeError(
            "Committed PB4-C1 protocol differs from deterministic pre-GPU "
            "derivation from the frozen PB4-C0 family"
        )
    return actual


def execute_with_evidence_buffers(
        runtime,
        shortlist,
        rule,
        qpos,
        *,
        live_records,
        restore_records,
        action_records):
    """Run PB4-C1 while retaining evidence if a later barrier blocks."""
    if live_records or restore_records or action_records:
        raise ValueError("PB4-C1 formal execution buffers must start empty")

    env = pb4.build_env(
        n_envs=int(runtime["replay"]["n_envs"]),
        n_steps_sub=int(runtime["replay"]["n_steps_sub"]),
        log_dir=pb4.official_log_dir(),
    )
    env.init_domain_randomization(**pb4.wrapping_args)
    build_state = env.scene.get_state()

    try:
        live, _, by_batch, n_intervals = (
            pb4._collect_live_histories_with_prefix_validity(
                env,
                runtime,
                shortlist,
                qpos,
                build_state,
            )
        )
        live_records.extend(
            pb4._revalidate_pairs_with_prefix(shortlist, live, rule)
        )
        pb4._live_barrier(live_records)

        sides = pb4.required_side_records(shortlist)
        side_output = {
            key: {
                "meta": row,
                "alignment": None,
                "live_history": live[key]["live_history"],
                "common_action_history": live[key]["common_action_history"],
                "snapshot_restore_errors_m": [],
                "repeats": [],
            }
            for key, row in sides.items()
        }

        snapshots, references = pb4._capture_snapshots(
            env,
            runtime,
            by_batch,
            qpos,
            build_state,
            n_intervals,
        )
        if len(sides) != 20:
            raise pb4.PB4Blocked(
                pb4.SNAPSHOT_FAILED,
                {
                    "failure_component": "snapshot_capture_count",
                    "expected_branch_count": 20,
                    "actual_branch_count": len(sides),
                    "action_evaluation_started": False,
                },
            )

        pb4._snapshot_barrier(
            env,
            runtime,
            by_batch,
            snapshots,
            references,
            side_output,
            restore_records,
        )

        pb4._evaluate_actions(
            env,
            runtime,
            by_batch,
            snapshots,
            qpos,
            n_intervals,
            action_records,
        )
    finally:
        env.stop()

    return live_records, restore_records, action_records


def map_blocked_verdict(base_verdict):
    if base_verdict not in PB4_BLOCKED_MAP:
        raise RuntimeError(f"Unexpected PB4 helper blocked verdict: {base_verdict}")
    return PB4_BLOCKED_MAP[base_verdict]


def enrich_audit(config, sources, audit):
    result = copy.deepcopy(audit)
    result["classification"] = "EXPLORATORY"
    result["formal_condition5_confirmed"] = False
    result["alpha_formal"] = sources["family_validation"]["alpha_formal"]
    result["winding_strata_source"] = config["control_rules"]["winding_strata_source"]
    return result


def write_outputs(
        config,
        sources,
        runtime,
        protocol,
        verdict,
        live_records,
        restore_records,
        action_records,
        *,
        audit=None,
        blocked=None):
    raw_root = Path(config["outputs"]["raw_root"])
    report_dir = REPO_ROOT / config["outputs"]["committed_report_dir"]
    raw_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    completed = verdict in (SIGNAL_FOUND, SIGNAL_NOT_FOUND)
    alpha = sources["family_validation"]["alpha_formal"]
    scientific = {
        "phase": "PB4-C1",
        "classification": "EXPLORATORY",
        "verdict": verdict,
        "formal_condition5_confirmed": False,
        "condition5_confirmation_allowed": False,
        "exploratory_signal_status": (
            "FOUND" if verdict == SIGNAL_FOUND
            else "NOT_FOUND" if verdict == SIGNAL_NOT_FOUND
            else "UNTESTED"
        ),
        "protocol": protocol,
        "frozen_alpha_formal": alpha,
        "live_pair_revalidation_records": live_records,
        "live_pair_barrier": {
            "valid": len(live_records) == 10 and all(bool(row.get("valid")) for row in live_records),
            "valid_pair_count": int(sum(bool(row.get("valid")) for row in live_records)),
            "required_pair_count": 10,
        },
        "snapshot_restore_records": restore_records,
        "snapshot_restore_barrier": {
            "valid": len(restore_records) == 60 and all(bool(row.get("valid")) for row in restore_records),
            "restore_record_count": len(restore_records),
            "required_restore_record_count": 60,
        },
        "action_evaluation_record_count": len(action_records),
        "expected_action_evaluation_record_count": 540,
        "control_relevance_status": "TESTED" if completed else "UNTESTED",
        "audit": audit,
        "blocked_details": blocked,
        "interpretation_boundary": {
            "same_cohort_reused_after_pb4_outcome_seen": True,
            "positive_result_is_confirmatory": False,
            "negative_result_stops_wrapping_control_relevance_route": verdict == SIGNAL_NOT_FOUND,
        },
        "boundaries": {
            "pair_drop": False,
            "pair_replacement": False,
            "alpha_recomputed": False,
            "alpha_rescaled": False,
            "qpos_targets_clipped": False,
            "action_library_modified_after_protocol": False,
            "deployable_sensor_test_started": False,
            "state_diff_training_started": False,
            "cfpm_training_started": False,
        },
    }

    (raw_root / "ACTION_EVALUATIONS.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "classification": "EXPLORATORY",
                "formal_condition5_confirmed": False,
                "alpha_formal": alpha,
                "records": action_records,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (raw_root / "AUDIT.json").write_text(
        json.dumps(scientific, indent=2) + "\n",
        encoding="utf-8",
    )

    evidence = {
        "phase_name": config["phase_name"],
        "verdict": verdict,
        "repository": {
            "starting_main_sha": config["provenance"]["starting_main_sha"],
            "ending_main_sha": pb4.git("rev-parse", "HEAD"),
            "dlolab_gitlink": sources["dlo_gitlink"],
        },
        "scientific": scientific,
    }
    (report_dir / "EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_dir / "PAIR_REGRET.json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "classification": "EXPLORATORY",
                "formal_condition5_confirmed": False,
                "alpha_formal": alpha,
                "audit": audit,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# PB4-C1 Feasibility-Normalized Signed Qpos-Target Exploration",
        "",
        f"Verdict: `{verdict}`",
        "",
        "**Classification:** `EXPLORATORY`",
        "",
        "**Formal Condition-5 confirmed:** `false`",
        "",
        f"**Frozen alpha_formal:** `{alpha:.17g}`",
        "",
        "## Barriers",
        "",
        f"- Live pair revalidation: {scientific['live_pair_barrier']['valid_pair_count']}/10",
        f"- Snapshot restore records: {scientific['snapshot_restore_barrier']['restore_record_count']}/60",
        f"- Action evaluation records: {len(action_records)}/540",
    ]
    if audit is not None:
        lines += [
            "",
            "## Exploratory control signal",
            "",
            f"- Passing pairs: {audit['passing_pair_count']}/10",
            f"- Passing PB4-C1 fresh-live winding strata: {audit['passing_winding_strata']}",
            "- Absolute regret floor: 0.05",
            "- Repeat-floor multiplier: 5",
        ]
    lines += ["", "## Stop rule", ""]
    if verdict == SIGNAL_FOUND:
        lines.append(
            "Exploratory signal found on the reused cohort. Stop. Do not claim "
            "Condition 5. The next allowed scientific step is an independent "
            "prospective confirmatory cohort using this exact frozen family."
        )
    elif verdict == SIGNAL_NOT_FOUND:
        lines.append(
            "Exploratory signal not found. Stop the Wrapping control-relevance "
            "route. Do not add new alpha values, horizons, adaptive scales, "
            "clipping, or hand-designed rescue actions."
        )
    else:
        lines.append(
            "Execution blocked before a complete exploratory result. Do not "
            "interpret this as a control-relevance negative."
        )
    lines.append("")
    (report_dir / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")
    return scientific


def run(config_path):
    config = load_json(config_path)
    sources = validate_sources(config)
    runtime = build_runtime_config(config, sources)
    protocol = load_and_validate_protocol(config, sources, runtime)

    live_records = []
    restore_records = []
    action_records = []

    try:
        execute_with_evidence_buffers(
            runtime,
            sources["shortlist"],
            sources["pb3r3_rule"],
            sources["qpos"],
            live_records=live_records,
            restore_records=restore_records,
            action_records=action_records,
        )

        audit = pb4.analyze_control_relevance(
            runtime,
            sources["shortlist"],
            action_records,
            live_records,
        )
        audit = enrich_audit(config, sources, audit)
        verdict = SIGNAL_FOUND if audit["confirmed"] else SIGNAL_NOT_FOUND

        pb4._save_score_npz(runtime, sources["shortlist"], action_records)
        scientific = write_outputs(
            config,
            sources,
            runtime,
            protocol,
            verdict,
            live_records,
            restore_records,
            action_records,
            audit=audit,
        )
    except pb4.PB4Blocked as exc:
        verdict = map_blocked_verdict(exc.verdict)
        scientific = write_outputs(
            config,
            sources,
            runtime,
            protocol,
            verdict,
            live_records,
            restore_records,
            action_records,
            blocked=exc.details,
        )

    print(f"verdict={scientific['verdict']}")
    return scientific


def validate_result_boundaries(verdict, scientific, action_records):
    audit = scientific.get("audit")
    live = scientific["live_pair_barrier"]
    snapshot = scientific["snapshot_restore_barrier"]
    blocked = scientific.get("blocked_details")

    if verdict in (SIGNAL_FOUND, SIGNAL_NOT_FOUND):
        if not live["valid"]:
            raise RuntimeError("Completed PB4-C1 result lacks live barrier")
        if not snapshot["valid"]:
            raise RuntimeError("Completed PB4-C1 result lacks snapshot barrier")
        if len(action_records) != 540:
            raise RuntimeError(
                f"Completed PB4-C1 requires 540 action records, got {len(action_records)}"
            )
        if audit is None:
            raise RuntimeError("Completed PB4-C1 result requires non-null audit")
        if blocked is not None:
            raise RuntimeError("Completed PB4-C1 cannot contain blocked details")
        return "complete"

    if verdict == LIVE_FAILED:
        if live["valid"]:
            raise RuntimeError("PB4-C1 LIVE_FAILED cannot have valid live barrier")
        if len(scientific["snapshot_restore_records"]) != 0:
            raise RuntimeError("PB4-C1 LIVE_FAILED cannot have snapshot records")
        if len(action_records) != 0 or audit is not None:
            raise RuntimeError("PB4-C1 LIVE_FAILED must be pre-action")
        if not blocked or blocked.get("failure_component") != "live_cohort_revalidation":
            raise RuntimeError("PB4-C1 LIVE_FAILED blocked_details mismatch")
        return "blocked_live"

    if verdict == SNAPSHOT_FAILED:
        if not live["valid"] or snapshot["valid"]:
            raise RuntimeError("PB4-C1 SNAPSHOT_FAILED barrier state invalid")
        if len(action_records) != 0 or audit is not None:
            raise RuntimeError("PB4-C1 SNAPSHOT_FAILED must be pre-action")
        if not blocked or blocked.get("failure_component") not in (
            "snapshot_restore_validation",
            "snapshot_capture_count",
        ):
            raise RuntimeError("PB4-C1 SNAPSHOT_FAILED blocked_details mismatch")
        return "blocked_snapshot"

    if verdict == ACTION_FAILED:
        if not live["valid"] or not snapshot["valid"]:
            raise RuntimeError("PB4-C1 ACTION_FAILED requires passed barriers")
        if len(action_records) > 540 or audit is not None:
            raise RuntimeError("PB4-C1 ACTION_FAILED boundary invalid")
        if not blocked:
            raise RuntimeError("PB4-C1 ACTION_FAILED requires blocked_details")
        return "blocked_action"

    raise RuntimeError(f"Unsupported PB4-C1 verdict: {verdict}")


def validate_complete_action_matrix(runtime, shortlist, action_records):
    action_ids = [row["action_id"] for row in runtime["action_library"]["actions"]]
    expected = set()
    for pair in shortlist["pairs"]:
        t = int(pair["time_index"])
        for rollout_key in ("rollout_a", "rollout_b"):
            rollout_id = int(pair[rollout_key])
            for action_id in action_ids:
                for repeat in range(3):
                    expected.add((rollout_id, t, action_id, repeat))

    actual = {
        (
            int(row["rollout_id"]),
            int(row["time_index"]),
            row["action_id"],
            int(row["repeat_index"]),
        )
        for row in action_records
    }
    if actual != expected:
        missing = list(expected - actual)[:10]
        extra = list(actual - expected)[:10]
        raise RuntimeError(
            "PB4-C1 action matrix key mismatch: "
            f"expected={len(expected)}, actual={len(actual)}, "
            f"missing_preview={missing}, extra_preview={extra}"
        )
    return len(actual)


def validate_results(config_path):
    config = load_json(config_path)
    sources = validate_sources(config)
    runtime = build_runtime_config(config, sources)
    protocol = load_and_validate_protocol(config, sources, runtime)

    raw_root = Path(config["outputs"]["raw_root"])
    report_dir = REPO_ROOT / config["outputs"]["committed_report_dir"]
    action_path = raw_root / "ACTION_EVALUATIONS.json"
    audit_path = raw_root / "AUDIT.json"
    evidence_path = report_dir / "EVIDENCE.json"
    regret_path = report_dir / "PAIR_REGRET.json"

    for path in (action_path, audit_path, evidence_path, regret_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    action_payload = load_json(action_path)
    raw_audit = load_json(audit_path)
    evidence = load_json(evidence_path)
    regret = load_json(regret_path)

    verdict = evidence["verdict"]
    scientific = evidence["scientific"]
    action_records = action_payload["records"]

    if action_payload.get("verdict") != verdict:
        raise RuntimeError("PB4-C1 raw/evidence verdict mismatch")
    if regret.get("verdict") != verdict:
        raise RuntimeError("PB4-C1 regret/evidence verdict mismatch")
    if raw_audit != scientific:
        raise RuntimeError("PB4-C1 raw AUDIT differs from committed evidence")
    if scientific["protocol"] != protocol:
        raise RuntimeError("PB4-C1 evidence protocol mismatch")

    for obj, label in (
        (protocol, "protocol"),
        (scientific, "evidence"),
        (regret, "PAIR_REGRET"),
        (action_payload, "ACTION_EVALUATIONS"),
    ):
        if obj.get("formal_condition5_confirmed") is not False:
            raise RuntimeError(
                f"PB4-C1 {label} must explicitly keep Condition 5 false"
            )

    alpha = sources["family_validation"]["alpha_formal"]
    if float(scientific["frozen_alpha_formal"]) != alpha:
        raise RuntimeError("PB4-C1 evidence alpha differs from frozen family")
    if float(regret["alpha_formal"]) != alpha:
        raise RuntimeError("PB4-C1 PAIR_REGRET alpha differs from frozen family")
    if float(action_payload["alpha_formal"]) != alpha:
        raise RuntimeError("PB4-C1 raw action alpha differs from frozen family")

    pb4.validate_action_record_scores(action_records, 20)
    mode = validate_result_boundaries(verdict, scientific, action_records)

    if mode == "complete":
        validate_complete_action_matrix(runtime, sources["shortlist"], action_records)
        expected_audit = pb4.analyze_control_relevance(
            runtime,
            sources["shortlist"],
            action_records,
            scientific["live_pair_revalidation_records"],
        )
        expected_audit = enrich_audit(config, sources, expected_audit)
        if scientific["audit"] != expected_audit:
            raise RuntimeError(
                "PB4-C1 evidence audit differs from deterministic recomputation"
            )
        if regret.get("audit") != expected_audit:
            raise RuntimeError(
                "PB4-C1 PAIR_REGRET differs from deterministic recomputation"
            )
        expected_verdict = SIGNAL_FOUND if expected_audit["confirmed"] else SIGNAL_NOT_FOUND
        if verdict != expected_verdict:
            raise RuntimeError(
                f"PB4-C1 verdict mismatch: {verdict} != {expected_verdict}"
            )
    else:
        if scientific.get("audit") is not None:
            raise RuntimeError("Blocked PB4-C1 evidence must have audit=null")
        if regret.get("audit") is not None:
            raise RuntimeError("Blocked PB4-C1 PAIR_REGRET must have audit=null")

    print("PB4-C1 result validation: PASS")
    print(f"verdict={verdict}")
    print(f"validation_mode={mode}")
    print(f"alpha_formal={alpha:.17g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "command",
        choices=("freeze-protocol", "run", "validate-results"),
    )
    args = parser.parse_args()

    if args.command == "freeze-protocol":
        freeze_protocol(args.config)
    elif args.command == "run":
        run(args.config)
    else:
        validate_results(args.config)


if __name__ == "__main__":
    main()
