"""PB3-B2 prospective cohort same-action causal audit.

Frozen order:
1. revalidate all 10 PB3-B1 live pairs;
2. capture and validate all 20 native snapshot restores;
3. only then execute any future suffix and the unchanged Gate 4.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.experiment3.dlolab_wrapping.pair_discovery_pb2c import (
    winding_integer_residual,
)
from scripts.experiment3.dlolab_wrapping.paths import REPO_ROOT, official_log_dir
from scripts.experiment3.dlolab_wrapping.preregister_alignment_pb3r3 import (
    live_pair_revalidation,
)
from scripts.experiment3.dlolab_wrapping.replay_winding_audit import (
    _dual_arm_command,
    build_env,
    load_json,
    normalize_qpos,
    sample_env,
    to_numpy,
)
from scripts.experiment3.dlolab_wrapping.snapshot_causal_audit_pb3 import (
    PB3Blocked,
    _jsonable,
    analyze_pairs,
    extract_env_sample,
    git,
    group_sides_by_batch_time,
    load_frozen_rollouts,
    required_side_records,
    save_trajectory_npz,
    seed_everything,
)
from utils.domain_randomization import wrapping_args


LIVE_FAILED = "PB3B2_LIVE_COHORT_REVALIDATION_FAILED"
SNAPSHOT_FAILED = "PB3B2_SNAPSHOT_RESTORE_FAILED"
SUFFIX_FAILED = "PB3B2_REPEAT_SUFFIX_INVALID"


def _parse_index(value):
    if isinstance(value, str):
        return [int(v) for v in value.split()]
    return [int(v) for v in value]


def validate_and_normalize_cohort(config, raw):
    cohort_path = REPO_ROOT / config["source"]["cohort"]
    evidence_path = REPO_ROOT / config["source"]["pb3b1_evidence"]
    rule_path = REPO_ROOT / config["source"]["pb3r3_alignment_rule"]
    for path in (cohort_path, evidence_path, rule_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    cohort = load_json(cohort_path)
    evidence = load_json(evidence_path)
    rule = load_json(rule_path)
    expected = config["source"]["expected_pb3b1_verdict"]
    if cohort.get("verdict") != expected or evidence.get("verdict") != expected:
        raise RuntimeError("PB3-B1 cohort/evidence verdict mismatch")
    if evidence["scientific"].get("cohort") != cohort:
        raise RuntimeError("Committed PB3-B1 cohort differs from evidence")
    if rule.get("verdict") != config["source"]["expected_pb3r3_verdict"]:
        raise RuntimeError("PB3-R3 rule verdict mismatch")

    selection = cohort.get("selection", {})
    pairs = cohort.get("pairs", [])
    used = [int(p[key]) for p in pairs for key in ("rollout_a", "rollout_b")]
    if len(pairs) != 10 or len(used) != 20 or len(set(used)) != 20:
        raise RuntimeError("PB3-B1 cohort must contain 10 pairs / 20 unique rollouts")
    if selection.get("accepted_pair_count") != 10:
        raise RuntimeError("PB3-B1 accepted pair count changed")
    if selection.get("unique_rollout_count") != 20:
        raise RuntimeError("PB3-B1 unique rollout count changed")
    if selection.get("pair_drop_allowed") or selection.get("pair_replacement_allowed"):
        raise RuntimeError("PB3-B1 cohort must forbid pair drop/replacement")
    if selection.get("future_information_used"):
        raise RuntimeError("PB3-B1 cohort used future information")
    if cohort.get("live_winding_strata_count", 0) < 2:
        raise RuntimeError("PB3-B1 cohort lost its two winding strata")
    if cohort.get("phase_boundary", {}).get("future_suffix_executed"):
        raise RuntimeError("PB3-B1 unexpectedly executed a future suffix")

    ids = [int(v) for v in raw["rollout_id"]]
    lookup = {rollout_id: index for index, rollout_id in enumerate(ids)}
    if len(lookup) != len(ids):
        raise RuntimeError("PB3-B1 raw rollout IDs are not unique")

    normalized = []
    for row in pairs:
        a = int(row["rollout_a"])
        b = int(row["rollout_b"])
        t = int(row["time_index"])
        if a not in lookup or b not in lookup:
            raise RuntimeError("PB3-B1 cohort rollout missing from raw data")
        for rollout_id, replay in ((a, row["replay_a"]), (b, row["replay_b"])):
            if rollout_id != int(replay["batch_index"]) * 32 + int(replay["env_index"]):
                raise RuntimeError("PB3-B1 replay locator mismatch")
            if int(replay["seed"]) != 123 + int(replay["batch_index"]):
                raise RuntimeError("PB3-B1 replay seed mismatch")

        ia = _parse_index(row["live_metrics"]["winding_index_a"])
        ib = _parse_index(row["live_metrics"]["winding_index_b"])
        turns_a = raw["signed_winding_turns"][lookup[a], t]
        turns_b = raw["signed_winding_turns"][lookup[b], t]
        residual = float(np.max(np.concatenate((
            winding_integer_residual(turns_a).reshape(-1),
            winding_integer_residual(turns_b).reshape(-1),
        ))))
        normalized.append({
            "pair_id": f"pb3b1_pair_{int(row['cohort_rank']):02d}",
            "source_rank": int(row["source_rank"]),
            "rollout_a": a,
            "rollout_b": b,
            "time_index": t,
            "winding_index_a": ia,
            "winding_index_b": ib,
            "differing_post_indices": [i for i, (x, y) in enumerate(zip(ia, ib)) if x != y],
            "visible_rope_history_chamfer_m": float(
                row["live_metrics"]["visible_rope_history_chamfer_m"]
            ),
            "pair_max_winding_integer_residual": residual,
            "replay_a": row["replay_a"],
            "replay_b": row["replay_b"],
        })

    shortlist = {
        "selection_rule": {
            "source": "frozen PB3-B1 prospective live cohort",
            "target_pair_count": 10,
            "max_rollout_use_count": 1,
            "future_information_used": False,
            "allow_pair_drop": False,
            "allow_pair_replacement": False,
        },
        "pairs": normalized,
    }
    validation = {
        "pb3b1_cohort_exactly_matches_evidence": True,
        "pair_count": 10,
        "unique_rollout_count": 20,
        "pair_drop_allowed": False,
        "pair_replacement_allowed": False,
        "live_winding_strata_count": int(cohort["live_winding_strata_count"]),
    }
    return shortlist, validation, rule


def validate_sources(config):
    raw_path = Path(config["source"]["pb3b1_raw_rollouts"])
    if not raw_path.is_file():
        raise FileNotFoundError(str(raw_path))
    raw = load_frozen_rollouts(raw_path)
    shortlist, validation, rule = validate_and_normalize_cohort(config, raw)

    if _jsonable(dict(wrapping_args)) != config["replay"]["expected_wrapping_args"]:
        raise RuntimeError("Pinned wrapping_args mismatch")
    if float(config["barriers"]["snapshot_restore_rope_max_abs_m"]) != float(
        rule["snapshot_restore_alignment"]["threshold_m"]
    ):
        raise RuntimeError("PB3-B2 snapshot threshold differs from frozen R3 rule")
    for key in (
        "absolute_effect_min_m",
        "repeat_floor_multiplier",
        "minimum_passing_pairs",
        "minimum_passing_winding_strata",
    ):
        if float(config["gate4"][key]) != float(rule["gate4"][key]):
            raise RuntimeError(f"PB3-B2 Gate 4 differs from frozen R3 rule at {key}")
    barriers = config["barriers"]
    if not barriers["require_all_10_live_pairs"] or not barriers["require_20_unique_branches"]:
        raise RuntimeError("PB3-B2 global barriers must remain enabled")
    if barriers["allow_pair_drop"] or barriers["allow_pair_replacement"]:
        raise RuntimeError("PB3-B2 must forbid pair drop/replacement")
    if barriers["targeted_frozen_alignment_is_gate"]:
        raise RuntimeError("PB3-B2 must not reintroduce historical targeted alignment")
    return raw, shortlist, validation, rule


def enforce_live_cohort_barrier(records):
    ids = [row["pair_id"] for row in records]
    failed = [row for row in records if not row.get("valid", False)]
    if len(records) != 10 or len(set(ids)) != 10 or failed:
        raise PB3Blocked(LIVE_FAILED, {
            "failure_component": "live_cohort_revalidation",
            "expected_pair_count": 10,
            "actual_pair_count": len(records),
            "unique_pair_count": len(set(ids)),
            "failed_pairs": failed,
            "causal_future_status": "UNTESTED",
            "snapshot_capture_count": 0,
            "future_suffix_executed": False,
        })


def enforce_snapshot_restore_barrier(records, repeat_count=3):
    keys = [(int(r["rollout_id"]), int(r["time_index"])) for r in records]
    counts = Counter(keys)
    failed = [row for row in records if not row.get("valid", False)]
    complete = len(counts) == 20 and all(v == repeat_count for v in counts.values())
    if len(records) != 20 * repeat_count or not complete or failed:
        raise PB3Blocked(SNAPSHOT_FAILED, {
            "failure_component": "snapshot_restore_validation",
            "expected_branch_count": 20,
            "actual_branch_count": len(counts),
            "expected_restore_records": 20 * repeat_count,
            "actual_restore_records": len(records),
            "failed_restores": failed,
            "causal_future_status": "UNTESTED",
            "future_suffix_executed": False,
        })


def _collect_live_histories(env, config, shortlist, qpos, build_state):
    sides = required_side_records(shortlist)
    groups = group_sides_by_batch_time(sides)
    by_batch = {}
    for (batch, time_index), members in groups.items():
        by_batch.setdefault(batch, {})[time_index] = members
    n_intervals = env.steps_interval // env._cmaes_n_steps_sub
    output = {key: {"meta": row} for key, row in sides.items()}

    for batch in sorted(by_batch):
        env.scene.reset(state=build_state)
        seed_everything(int(config["replay"]["base_seed"]) + int(batch))
        env.use_qpos = True
        env.reset()
        times = sorted(by_batch[batch])
        history_steps = {step for t in times for step in (t - 2, t - 1, t)}
        samples = {}
        for step in range(1, max(times) + 1):
            _dual_arm_command(env, qpos[step])
            for _ in range(n_intervals):
                env.scene.step()
            if step in history_steps:
                samples[step] = sample_env(env)
        for t in times:
            for key, row in by_batch[batch][t]:
                output[key]["live_history"] = [
                    extract_env_sample(samples[step], int(row["env_index"]))
                    for step in (t - 2, t - 1, t)
                ]
                output[key]["common_action_history"] = np.asarray(
                    qpos[t - 2:t + 1]
                ).copy()
    return output, groups, by_batch, n_intervals


def _revalidate_pairs(shortlist, live, rule):
    records = []
    for pair in shortlist["pairs"]:
        t = int(pair["time_index"])
        key_a = (int(pair["rollout_a"]), t)
        key_b = (int(pair["rollout_b"]), t)
        result = live_pair_revalidation(
            live[key_a]["live_history"],
            live[key_b]["live_history"],
            rule,
            same_time=True,
            common_action_history_equal=np.array_equal(
                live[key_a]["common_action_history"],
                live[key_b]["common_action_history"],
            ),
        )
        records.append({
            "pair_id": pair["pair_id"],
            "rollout_a": pair["rollout_a"],
            "rollout_b": pair["rollout_b"],
            "time_index": t,
            **result,
        })
    return records


def _capture_snapshots(env, config, by_batch, qpos, build_state, n_intervals):
    snapshots = {}
    references = {}
    for batch in sorted(by_batch):
        env.scene.reset(state=build_state)
        seed_everything(int(config["replay"]["base_seed"]) + int(batch))
        env.use_qpos = True
        env.reset()
        target_times = sorted(by_batch[batch])
        for step in range(1, max(target_times) + 1):
            _dual_arm_command(env, qpos[step])
            for _ in range(n_intervals):
                env.scene.step()
            if step in by_batch[batch]:
                group = (batch, step)
                references[group] = sample_env(env)
                snapshots[group] = env.scene.get_state()
    return snapshots, references


def _validate_all_snapshot_restores(
        env, config, by_batch, snapshots, references, side_output, records):
    repeats = int(config["replay"]["snapshot_repeat_count"])
    threshold = float(config["barriers"]["snapshot_restore_rope_max_abs_m"])
    for batch in sorted(by_batch):
        for t in sorted(by_batch[batch]):
            group = (batch, t)
            members = by_batch[batch][t]
            for repeat in range(repeats):
                env.scene.reset(state=snapshots[group])
                restored = sample_env(env)
                for key, row in members:
                    env_index = int(row["env_index"])
                    reference_rope = np.asarray(references[group]["rope_xyz"][env_index])
                    restored_rope = np.asarray(restored["rope_xyz"][env_index])
                    error = float(np.max(np.abs(restored_rope - reference_rope)))
                    side_output[key]["snapshot_restore_errors_m"].append(error)
                    records.append({
                        "rollout_id": int(row["rollout_id"]),
                        "time_index": int(t),
                        "repeat_index": repeat,
                        "rope_max_abs_m": error,
                        "threshold_m": threshold,
                        "valid": bool(error <= threshold),
                    })
    enforce_snapshot_restore_barrier(records, repeats)
    return records


def _run_future_suffixes(
        env, config, by_batch, snapshots, qpos, n_intervals, side_output):
    horizons = [int(v) for v in config["replay"]["horizons"]]
    max_horizon = max(horizons)
    repeats = int(config["replay"]["snapshot_repeat_count"])
    stretch_limit = float(config["replay"]["official_stretch_ratio_limit"])
    for batch in sorted(by_batch):
        for t in sorted(by_batch[batch]):
            group = (batch, t)
            members = by_batch[batch][t]
            for repeat in range(repeats):
                env.scene.reset(state=snapshots[group])
                restored = sample_env(env)
                stores = {key: {0: extract_env_sample(restored, int(row["env_index"]))}
                          for key, row in members}
                valid = {key: True for key, _ in members}
                for relative in range(1, max_horizon + 1):
                    command_index = int(t) + relative
                    if command_index >= len(qpos):
                        raise RuntimeError("PB3-B2 horizon exceeds official qpos")
                    _dual_arm_command(env, qpos[command_index])
                    for _ in range(n_intervals):
                        env.scene.step()
                    rope = np.asarray(env.rope.get_all_verts())
                    distance = to_numpy(env.rope.get_geodesic_distance(
                        env.control_idx[0], env.control_idx[1]
                    )).reshape(env.n_envs)
                    stretch = distance / to_numpy(env.control_dist_init).reshape(env.n_envs)
                    for key, row in members:
                        env_index = int(row["env_index"])
                        if np.isnan(rope[env_index]).any() or stretch[env_index] > stretch_limit:
                            valid[key] = False
                    if relative in horizons:
                        sampled = sample_env(env)
                        for key, row in members:
                            stores[key][relative] = extract_env_sample(
                                sampled, int(row["env_index"])
                            )
                failed = [key for key in valid if not valid[key]]
                if failed:
                    raise PB3Blocked(SUFFIX_FAILED, {
                        "failure_component": "repeat_suffix_validity",
                        "failed_branch_keys": [list(key) for key in failed],
                        "repeat_index": repeat,
                        "future_suffix_executed": True,
                    })
                for key, _ in members:
                    side_output[key]["repeats"].append(stores[key])


def execute(config, shortlist, qpos, rule):
    return execute_with_evidence(config, shortlist, qpos, rule)


def execute_with_evidence(
        config, shortlist, qpos, rule, *, live_records=None,
        restore_records=None, execution_state=None):
    env = build_env(
        n_envs=int(config["replay"]["n_envs"]),
        n_steps_sub=int(config["replay"]["n_steps_sub"]),
        log_dir=official_log_dir(),
    )
    env.init_domain_randomization(**wrapping_args)
    build_state = env.scene.get_state()
    live_records = [] if live_records is None else live_records
    restore_records = [] if restore_records is None else restore_records
    execution_state = {} if execution_state is None else execution_state
    side_output = None
    snapshot_count = 0
    try:
        live, _, by_batch, n_intervals = _collect_live_histories(
            env, config, shortlist, qpos, build_state
        )
        live_records.extend(_revalidate_pairs(shortlist, live, rule))
        enforce_live_cohort_barrier(live_records)

        sides = required_side_records(shortlist)
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
        snapshots, references = _capture_snapshots(
            env, config, by_batch, qpos, build_state, n_intervals
        )
        snapshot_count = len(sides)
        execution_state["snapshot_capture_count"] = snapshot_count
        _validate_all_snapshot_restores(
            env, config, by_batch, snapshots, references, side_output,
            restore_records,
        )
        _run_future_suffixes(
            env, config, by_batch, snapshots, qpos, n_intervals, side_output
        )
    finally:
        env.stop()
    return side_output, live_records, restore_records, snapshot_count


def write_outputs(
        config, shortlist, validation, verdict, live_records, restore_records,
        snapshot_count, *, audit=None, trajectories_path=None, blocked=None):
    raw_root = Path(config["outputs"]["raw_root"])
    report_dir = REPO_ROOT / config["outputs"]["committed_report_dir"]
    raw_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    future = verdict in (
        "PB3_CAUSAL_FUTURE_BIFURCATION_CONFIRMED",
        "PB3_CAUSAL_FUTURE_BIFURCATION_NOT_CONFIRMED",
        SUFFIX_FAILED,
    )
    result = {
        "phase": "PB3-B2",
        "verdict": verdict,
        "causal_future_status": "TESTED" if audit is not None else "UNTESTED",
        "cohort_validation": validation,
        "live_pair_revalidation_records": live_records,
        "live_pair_barrier": {
            "valid": len(live_records) == 10 and all(r.get("valid", False) for r in live_records),
            "valid_pair_count": sum(bool(r.get("valid", False)) for r in live_records),
            "required_pair_count": 10,
        },
        "snapshot_capture_count": int(snapshot_count),
        "snapshot_restore_records": restore_records,
        "snapshot_restore_barrier": {
            "valid": len(restore_records) == 60 and all(r.get("valid", False) for r in restore_records),
            "valid_branch_count": len({
                (r["rollout_id"], r["time_index"])
                for r in restore_records if r.get("valid", False)
            }),
            "required_branch_count": 20,
            "restore_record_count": len(restore_records),
        },
        "future_suffix_executed": bool(future),
        "horizons": config["replay"]["horizons"],
        "snapshot_repeat_count": config["replay"]["snapshot_repeat_count"],
        "gate4": config["gate4"],
        "audit": audit,
        "blocked_details": blocked,
        "trajectories_path": None if trajectories_path is None else str(trajectories_path),
        "boundaries": {
            "pair_drop": False,
            "pair_replacement": False,
            "targeted_frozen_alignment_used_as_gate": False,
            "pb4_started": False,
            "training_started": False,
        },
    }
    (raw_root / "AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    evidence = {
        "phase_name": config["phase_name"],
        "verdict": verdict,
        "repository": {
            "starting_main_sha": config["provenance"]["starting_main_sha"],
            "ending_main_sha": git("rev-parse", "HEAD"),
            "dlolab_gitlink": git("rev-parse", "HEAD:external/dlo-lab"),
        },
        "scientific": result,
    }
    (report_dir / "EVIDENCE.json").write_text(json.dumps(evidence, indent=2) + "\n")
    pair_metrics = {"verdict": verdict, "pair_metrics": [] if audit is None else audit["pairs"]}
    (report_dir / "PAIR_METRICS.json").write_text(json.dumps(pair_metrics, indent=2) + "\n")

    lines = [
        "# PB3-B2 Prospective Wrapping Causal Audit",
        "",
        f"Verdict: `{verdict}`",
        "",
        f"- Frozen cohort: 10 pairs / 20 unique rollouts / no replacement",
        f"- Live pair revalidation: {result['live_pair_barrier']['valid_pair_count']}/10",
        f"- Snapshot capture count: {snapshot_count}/20",
        f"- Snapshot restore valid branches: {result['snapshot_restore_barrier']['valid_branch_count']}/20",
        f"- Future suffix executed: {future}",
        f"- Causal future status: {result['causal_future_status']}",
    ]
    if audit is not None:
        lines += [
            f"- Gate-4 passing pairs: {audit['passing_pair_count']}/{audit['pair_count']}",
            f"- Gate-4 passing winding strata: {audit['passing_winding_strata']}",
        ]
    lines += [
        "",
        "## Boundary",
        "",
        "No pair was dropped or replaced. PB3-B2 does not establish control relevance, deployable sensing, or model performance.",
        "",
    ]
    (report_dir / "RESULT.md").write_text("\n".join(lines))
    return result


def validate_only(config_path):
    config = load_json(config_path)
    raw, shortlist, validation, _ = validate_sources(config)
    qpos = normalize_qpos(np.load(official_log_dir() / "best_qpos.npy"))
    if not np.array_equal(qpos, raw["common_qpos_replay"]):
        raise RuntimeError("PB3-B2 official qpos differs from PB3-B1 frozen replay")
    print("PB3-B2 source validation: PASS")
    print(f"pairs={len(shortlist['pairs'])} unique_rollouts={validation['unique_rollout_count']}")


def run(config_path):
    config = load_json(config_path)
    raw, shortlist, validation, rule = validate_sources(config)
    qpos = normalize_qpos(np.load(official_log_dir() / "best_qpos.npy"))
    if not np.array_equal(qpos, raw["common_qpos_replay"]):
        raise RuntimeError("PB3-B2 official qpos differs from PB3-B1 frozen replay")
    live_records = []
    restore_records = []
    execution_state = {"snapshot_capture_count": 0}
    try:
        side_output, live_records, restore_records, snapshot_count = execute_with_evidence(
            config, shortlist, qpos, rule,
            live_records=live_records,
            restore_records=restore_records,
            execution_state=execution_state,
        )
        audit = analyze_pairs(config, shortlist, side_output)
        trajectories = save_trajectory_npz(config, shortlist, side_output)
        verdict = (
            "PB3_CAUSAL_FUTURE_BIFURCATION_CONFIRMED"
            if audit["confirmed"]
            else "PB3_CAUSAL_FUTURE_BIFURCATION_NOT_CONFIRMED"
        )
        result = write_outputs(
            config, shortlist, validation, verdict, live_records, restore_records,
            snapshot_count, audit=audit, trajectories_path=trajectories
        )
    except PB3Blocked as exc:
        snapshot_count = int(execution_state["snapshot_capture_count"])
        result = write_outputs(
            config, shortlist, validation, exc.verdict, live_records,
            restore_records, snapshot_count, blocked=exc.details
        )
    print(f"verdict={result['verdict']}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_only(args.config)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
