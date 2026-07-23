from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_stagex_resume1_import_crossfit_recovery as rx
from ccda_phase3 import phase314b_r258_stagex_tail_robust_nested_oof as sx


def metric(overall=0.8, accepted=0.8, acceptance=0.8, positive=0.7, relative=0.1, adverse=0.1, cvar=0.9, brier=True):
    return {
        "acceptance_rate": acceptance,
        "overall_mse_ratio": overall,
        "accepted_row_mse_ratio": accepted,
        "positive_distance_reduction_rate": positive,
        "relative_distance_reduction_mean": relative,
        "adverse_sse_mass": adverse,
        "risk_brier_nonworse": brier,
        "group_tail": {"worst_fraction_cvar_mse_ratio": cvar},
    }


def good_records():
    return {t: metric() for t in sx.LOCKED_TIMESTEPS}


def baseline_records():
    return {
        t: metric(overall=0.9, accepted=0.9, adverse=0.2, cvar=1.1)
        for t in sx.LOCKED_TIMESTEPS
    }


def selections(eligible=True, ids=None):
    default = sx.policy_population()[0].policy_id
    ids = [default] * 6 if ids is None else ids
    result = []
    for value in ids:
        policy = next(p for p in sx.policy_population() if p.policy_id == value)
        result.append(
            {
                "selected_policy": sx.asdict(policy),
                "selected_policy_id": value,
                "selected_policy_inner_eligible": eligible,
            }
        )
    return result


def modal(support=6):
    policy = sx.policy_population()[0]
    return {
        "policy": sx.asdict(policy),
        "policy_id": policy.policy_id,
        "support_count": support,
        "counts": {policy.policy_id: support},
    }


def full_selection(eligible=True):
    policy = sx.policy_population()[0]
    return {
        "selected_policy": sx.asdict(policy),
        "selected_policy_id": policy.policy_id,
        "selected_policy_inner_eligible": eligible,
    }


def test_entrypoint_defect_detects_import_before_main():
    source = "from pathlib import Path\nfrom ccda_phase3 import x\nif __name__ == '__main__':\n    main()\n"
    result = rx.entrypoint_import_defect(source)
    assert result["failure_precedes_controller_main"]
    assert result["sys_path_bootstrap_before_import"] is False


def test_entrypoint_defect_rejects_bootstrap():
    source = "import sys\nsys.path.insert(0, '/x')\nfrom ccda_phase3 import x\nif __name__ == '__main__':\n    main()\n"
    with pytest.raises(rx.StageXError):
        rx.entrypoint_import_defect(source)


def test_child_environment_prepends_root(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/a", str(tmp_path)]))
    env = rx.child_environment(tmp_path)
    assert env["PYTHONPATH"].split(os.pathsep) == [str(tmp_path.resolve()), "/a"]


def test_child_environment_handles_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert rx.child_environment(tmp_path)["PYTHONPATH"] == str(tmp_path.resolve())


def clean_generation():
    return {
        "length_log_z_element_mismatch_count": 0,
        "aligned_upper_element_failure_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
        "candidate_finalized_without_holdout_target": True,
        "internal_scale_attempt_count": 7,
    }


def test_clean_generation_passes():
    rx.require_clean_generation(clean_generation(), label="x")


@pytest.mark.parametrize(
    "field",
    [
        "length_log_z_element_mismatch_count",
        "aligned_upper_element_failure_count",
        "strict_pass_aligned_fail_row_count",
    ],
)
def test_clean_generation_rejects_gate_failure(field):
    value = clean_generation(); value[field] = 1
    with pytest.raises(rx.StageXError):
        rx.require_clean_generation(value, label="x")


def test_clean_generation_rejects_target_finalization_change():
    value = clean_generation(); value["candidate_finalized_without_holdout_target"] = False
    with pytest.raises(rx.StageXError):
        rx.require_clean_generation(value, label="x")


def test_clean_generation_rejects_scale_count_change():
    value = clean_generation(); value["internal_scale_attempt_count"] = 6
    with pytest.raises(rx.StageXError):
        rx.require_clean_generation(value, label="x")


def test_checked_risk_model_accepts_constant(monkeypatch):
    monkeypatch.setattr(sx, "fit_risk_model", lambda *a, **k: {"mode":"constant", "prevalence":0.5})
    assert rx.fit_risk_model_checked(np.zeros((2,1)), np.zeros(2), np.ones(2,bool), sx.StageXSpec())["mode"] == "constant"


def test_checked_risk_model_rejects_nonconvergence(monkeypatch):
    monkeypatch.setattr(sx, "fit_risk_model", lambda *a, **k: {"coefficient":np.zeros(2), "converged":False})
    with pytest.raises(rx.StageXError):
        rx.fit_risk_model_checked(np.zeros((2,1)), np.zeros(2), np.ones(2,bool), sx.StageXSpec())


def test_checked_risk_model_accepts_converged(monkeypatch):
    monkeypatch.setattr(sx, "fit_risk_model", lambda *a, **k: {"coefficient":np.zeros(2), "converged":True})
    assert rx.fit_risk_model_checked(np.zeros((2,1)), np.zeros(2), np.ones(2,bool), sx.StageXSpec())["converged"]


def variable_arrays(rows=12):
    control = np.zeros((rows, 1, 1), dtype=np.float32)
    target = np.ones_like(control)
    candidate = np.full_like(control, 0.5)
    scale = np.ones(rows)
    risk = np.linspace(0, 1, rows)
    thresholds = np.asarray([0.25] * (rows//2) + [0.75] * (rows-rows//2))
    groups = np.asarray([f"g{i//2}" for i in range(rows)])
    constant = np.full(rows, 0.5)
    return control, target, candidate, scale, risk, thresholds, groups, constant


def test_variable_threshold_policy_uses_per_row_threshold():
    args = variable_arrays()
    result = rx.evaluate_variable_threshold_policy(
        args[0],args[2],args[3],args[4],args[5],args[1],args[6],args[7],sx.StageXSpec()
    )
    expected = (args[4] <= args[5])
    assert result["selected_row_count"] == int(np.sum(expected))


def test_variable_threshold_policy_rejects_shape():
    args = variable_arrays()
    with pytest.raises(rx.StageXError):
        rx.evaluate_variable_threshold_policy(
            args[0],args[2],args[3],args[4][:-1],args[5],args[1],args[6],args[7],sx.StageXSpec()
        )


def test_variable_threshold_policy_no_acceptance():
    args = list(variable_arrays()); args[4] = np.ones(12); args[5] = np.zeros(12)
    result = rx.evaluate_variable_threshold_policy(
        args[0],args[2],args[3],args[4],args[5],args[1],args[6],args[7],sx.StageXSpec()
    )
    assert result["acceptance_rate"] == 0.0
    assert result["overall_mse_ratio"] == 1.0


def synthetic_context(rows=12):
    return {"objective_control_predictions": {t: np.zeros((rows,1,1),np.float32) for t in sx.LOCKED_TIMESTEPS}}


def synthetic_outer_outputs(ids, rows=12):
    outputs = {}
    fold_size = rows // 6
    for fold in range(6):
        indices = np.arange(fold*fold_size,(fold+1)*fold_size)
        outputs[fold] = {}
        for t in sx.LOCKED_TIMESTEPS:
            outputs[fold][t] = {}
            for shrink in sx.DIRECTION_SHRINKAGES:
                candidate = np.full((indices.size,1,1), 0.2 + 0.2*shrink, np.float32)
                outputs[fold][t][shrink] = {
                    "indices": indices,
                    "candidate": candidate,
                    "selected_scale": np.ones(indices.size),
                    "risk_probability": np.zeros(indices.size),
                    "risk_constant_probability": np.full(indices.size,.5),
                    "gate_counts": {
                        "length_log_z_element_mismatch_count":0,
                        "aligned_upper_element_failure_count":0,
                        "strict_pass_aligned_fail_row_count":0,
                    },
                }
    return outputs


def test_stitch_fold_selected_uses_each_fold_policy(monkeypatch):
    rows=12; context=synthetic_context(rows)
    ids=[]
    for i in range(6):
        ids.append(sx.TailPolicy(0.5 if i<3 else 1.0, 1.0).policy_id)
    sel=selections(ids=ids)
    records,gates=rx.stitch_fold_selected_procedure(
        context=context, outer_outputs=synthetic_outer_outputs(ids,rows), outer_selections=sel,
        target=np.ones((rows,1,1),np.float32), groups=np.asarray([f"g{i}" for i in range(rows)]), spec=sx.StageXSpec()
    )
    assert set(records)==set(sx.LOCKED_TIMESTEPS)
    assert all(all(v==0 for v in gates[str(t)].values()) for t in sx.LOCKED_TIMESTEPS)


def test_stitch_rejects_overlapping_rows():
    rows=12; outputs=synthetic_outer_outputs([],rows)
    outputs[1][10][0.5]["indices"] = outputs[0][10][0.5]["indices"]
    with pytest.raises(rx.StageXError):
        rx.stitch_fold_selected_procedure(
            context=synthetic_context(rows), outer_outputs=outputs, outer_selections=selections(),
            target=np.ones((rows,1,1),np.float32), groups=np.asarray([f"g{i}" for i in range(rows)]), spec=sx.StageXSpec()
        )


def test_fixed_policy_surface_has_all_policies():
    rows=12
    records,base=rx.fixed_policy_oof_surface(
        context=synthetic_context(rows), outer_outputs=synthetic_outer_outputs([],rows),
        target=np.ones((rows,1,1),np.float32), groups=np.asarray([f"g{i}" for i in range(rows)]), spec=sx.StageXSpec()
    )
    assert set(records)=={p.policy_id for p in sx.policy_population()}
    assert set(base)==set(sx.LOCKED_TIMESTEPS)


def test_classify_ready_uses_procedure_and_full_selection():
    result=rx.classify_resume1(
        procedure_records=good_records(), baseline_records=baseline_records(),
        outer_selections=selections(), modal=modal(), full_selection=full_selection(), spec=sx.StageXSpec()
    )
    assert result["scientific_status"]=="READY"


def test_classify_blocks_outer_crossfit_failure():
    records=good_records(); records[10]=metric(overall=1.1)
    result=rx.classify_resume1(
        procedure_records=records, baseline_records=baseline_records(), outer_selections=selections(),
        modal=modal(), full_selection=full_selection(), spec=sx.StageXSpec()
    )
    assert result["primary_failure_locus"]=="outer_crossfit_tail_fidelity_failure"


def test_classify_blocks_inner_instability():
    result=rx.classify_resume1(
        procedure_records=good_records(), baseline_records=baseline_records(), outer_selections=selections(False),
        modal=modal(), full_selection=full_selection(), spec=sx.StageXSpec()
    )
    assert result["primary_failure_locus"]=="inner_selection_instability"


def test_classify_blocks_modal_instability():
    result=rx.classify_resume1(
        procedure_records=good_records(), baseline_records=baseline_records(), outer_selections=selections(),
        modal=modal(3), full_selection=full_selection(), spec=sx.StageXSpec()
    )
    assert result["primary_failure_locus"]=="policy_stability_failure"


def test_classify_blocks_fixed_policy_failure():
    result=rx.classify_resume1(
        procedure_records=good_records(), baseline_records=baseline_records(), outer_selections=selections(),
        modal=modal(), full_selection=full_selection(False), spec=sx.StageXSpec()
    )
    assert result["primary_failure_locus"]=="full_objective_oof_policy_failure"


def worker_payload(status="BLOCKED"):
    payload={
        "schema":rx.WORKER_SCHEMA,"execution_verdict":"PASS","scientific_status":status,
        "selection_holdout_evaluation_count_added":0,"cumulative_selection_holdout_evaluation_count":1,
        "selected_configuration":None,"rerun_authorized":False,
        "outer_crossfit_fold_selected_policy_records":{"10":{},"25":{},"50":{}},
        "train_only_recommendation":None if status=="BLOCKED" else {"x":1},
    }
    payload.update({key:False for key in rx.FALSE_BOUNDARIES})
    payload["scientific_result_sha256"]=rx.sha256_bytes(rx.stable_json_bytes(payload))
    return payload


def test_worker_payload_validates_blocked():
    rx.validate_worker_payload(worker_payload())


def test_worker_payload_validates_ready():
    rx.validate_worker_payload(worker_payload("READY"))


def test_worker_payload_rejects_hash():
    value=worker_payload(); value["scientific_result_sha256"]="0"*64
    with pytest.raises(rx.StageXError): rx.validate_worker_payload(value)


def test_worker_payload_rejects_holdout_access():
    value=worker_payload(); value["selection_holdout_evaluation_count_added"]=1
    value["scientific_result_sha256"]=rx.sha256_bytes(rx.stable_json_bytes({k:v for k,v in value.items() if k!="scientific_result_sha256"}))
    with pytest.raises(rx.StageXError): rx.validate_worker_payload(value)


def test_blocked_report_preserves_boundary():
    value=rx.blocked_report(None,RuntimeError("x"))
    assert value["selection_holdout_evaluation_count_added"]==0
    assert value["rerun_authorized"] is False


def test_implementation_path_population_is_add_only():
    assert len(rx.IMPLEMENTATION_PATHS)==4
    assert all(status=="A" for status,_ in rx.IMPLEMENTATION_PATHS)


def test_original_source_population_is_bound():
    assert set(rx.ORIGINAL_SOURCE_SHA256)=={path for _,path in rx.ORIGINAL_IMPLEMENTATION_PATHS}
