from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pytest
from ccda_phase3 import phase314b_r258_stagex_tail_robust_nested_oof as sx


def arrays(rows=12):
    control = np.zeros((rows, 4, 6), dtype=np.float32)
    target = np.ones_like(control)
    candidate = np.full_like(control, 0.8)
    scale = np.ones(rows, dtype=np.float64)
    risk = np.linspace(0.0, 1.0, rows)
    groups = np.asarray([f'g{i//2}' for i in range(rows)])
    direction = np.full_like(control, 0.2, dtype=np.float64)
    return control, target, candidate, scale, risk, groups, direction


def metric(overall=0.8, accepted=0.8, acceptance=0.8, positive=0.7, relative=0.1, adverse=0.1, cvar=0.9, brier=True):
    return {
        'acceptance_rate': acceptance, 'overall_mse_ratio': overall,
        'accepted_row_mse_ratio': accepted,
        'positive_distance_reduction_rate': positive,
        'relative_distance_reduction_mean': relative,
        'adverse_sse_mass': adverse, 'risk_brier_nonworse': brier,
        'group_tail': {'worst_fraction_cvar_mse_ratio': cvar},
    }


def baseline():
    return {t: metric(overall=0.9, accepted=0.9, adverse=0.2, cvar=1.1) for t in sx.LOCKED_TIMESTEPS}


def good_records():
    return {t: metric() for t in sx.LOCKED_TIMESTEPS}


@pytest.mark.parametrize('policy', sx.policy_population())
def test_policy_population_members_validate(policy):
    policy.validate()


def test_policy_population_size_and_unique():
    values = sx.policy_population(); assert len(values) == 12; assert len({x.policy_id for x in values}) == 12


def test_policy_rejects_invalid_shrinkage():
    with pytest.raises(sx.StageXError): sx.TailPolicy(0.6, 0.5).validate()


def test_policy_rejects_invalid_threshold():
    with pytest.raises(sx.StageXError): sx.TailPolicy(0.5, 0.6).validate()


def test_spec_validates():
    sx.StageXSpec().validate()


def test_spec_rejects_fold_change():
    with pytest.raises(sx.StageXError): sx.StageXSpec(outer_folds=5).validate()


def test_stable_json_order():
    assert sx.stable_json_bytes({'b': 2, 'a': 1}) == sx.stable_json_bytes({'a': 1, 'b': 2})


def test_sha_array_binds_shape():
    a = np.arange(6, dtype=np.float64); assert sx.sha256_array(a) != sx.sha256_array(a.reshape(2, 3))


def test_compact_descriptor_shape():
    c, _, q, s, _, _, d = arrays(); x = sx.compact_risk_descriptors(c, d, q, s); assert x.shape == (12, 17)


def test_compact_descriptor_rejects_shape_change():
    c, _, q, s, _, _, d = arrays()
    with pytest.raises(sx.StageXError): sx.compact_risk_descriptors(c, d[:, :3], q, s)


def test_risk_model_predicts_probabilities():
    x = np.column_stack([np.arange(20), np.arange(20) ** 2]).astype(float); y = (np.arange(20) > 9).astype(float)
    model = sx.fit_risk_model(x, y, np.ones(20, bool), sx.StageXSpec()); p = sx.predict_risk(model, x)
    assert p.shape == (20,); assert np.all((p >= 0) & (p <= 1))


def test_risk_model_uses_constant_for_small_population():
    model = sx.fit_risk_model(np.zeros((7, 2)), np.zeros(7), np.ones(7, bool), sx.StageXSpec()); assert model['mode'] == 'constant'


def test_row_squared_error():
    c, t, _, _, _, _, _ = arrays(2); np.testing.assert_allclose(sx.row_squared_error(c, t), 24.0)


def test_apply_policy_abstains_high_risk():
    c, _, q, s, r, _, _ = arrays(4); result = sx.apply_policy(c, q, s, r, 0.5)
    assert result['kept_mask'].tolist() == [True, True, False, False]


def test_group_tail_metrics():
    result = sx.group_tail_metrics(np.ones(6), np.asarray([1, 1, 2, 2, 3, 3]), ['a','a','b','b','c','c'], .2)
    assert result['group_count'] == 3; assert result['worst_group_mse_ratio'] == 3.0


def test_evaluate_policy_improves():
    c, t, q, s, _, g, _ = arrays(); result = sx.evaluate_policy(c, q, s, np.zeros(12), t, g, 1.0, None, sx.StageXSpec())
    assert result['overall_mse_ratio'] < 1; assert result['acceptance_rate'] == 1.0


def test_evaluate_policy_no_acceptance():
    c, t, q, s, _, g, _ = arrays(); result = sx.evaluate_policy(c, q, s, np.ones(12), t, g, 0.0, None, sx.StageXSpec())
    assert result['acceptance_rate'] == 0.0; assert result['overall_mse_ratio'] == 1.0


def test_policy_eligibility_passes_good_records():
    assert sx.policy_is_eligible(good_records(), baseline(), sx.StageXSpec())['all_timesteps_pass']


def test_policy_eligibility_rejects_low_acceptance():
    records = good_records(); records[10] = metric(acceptance=.4)
    assert not sx.policy_is_eligible(records, baseline(), sx.StageXSpec())['all_timesteps_pass']


def test_joint_selection_prefers_eligible():
    records = {p.policy_id: {t: metric(overall=.95, accepted=.95, adverse=.19, cvar=1.05) for t in sx.LOCKED_TIMESTEPS} for p in sx.policy_population()}
    best = sx.policy_population()[3]; records[best.policy_id] = good_records()
    selected = sx.select_joint_policy(records, baseline(), sx.StageXSpec()); assert selected['selected_policy_id'] == best.policy_id


def test_joint_selection_marks_diagnostic_fallback():
    records = {p.policy_id: {t: metric(overall=1.1, accepted=1.1, adverse=.3, cvar=1.2) for t in sx.LOCKED_TIMESTEPS} for p in sx.policy_population()}
    selected = sx.select_joint_policy(records, baseline(), sx.StageXSpec()); assert selected['diagnostic_fallback_used']


def test_modal_policy_counts_support():
    pid = sx.policy_population()[0].policy_id; result = sx.modal_policy([pid, pid, pid, pid, sx.policy_population()[1].policy_id, sx.policy_population()[2].policy_id])
    assert result['policy_id'] == pid; assert result['support_count'] == 4


def test_modal_policy_rejects_wrong_count():
    with pytest.raises(sx.StageXError): sx.modal_policy([sx.policy_population()[0].policy_id])


def selections(eligible=True):
    pid = sx.policy_population()[0].policy_id
    return [{'selected_policy_id': pid, 'selected_policy_inner_eligible': eligible} for _ in range(6)]


def final_policy(support=6):
    p = sx.policy_population()[0]
    return {'policy': sx.asdict(p), 'policy_id': p.policy_id, 'support_count': support, 'counts': {p.policy_id: support}}


def test_classification_ready():
    result = sx.classify_stagex(good_records(), baseline(), selections(), final_policy(), sx.StageXSpec()); assert result['scientific_status'] == 'READY'


def test_classification_inner_instability():
    result = sx.classify_stagex(good_records(), baseline(), selections(False), final_policy(), sx.StageXSpec()); assert result['primary_failure_locus'] == 'inner_selection_instability'


def test_classification_outer_instability():
    result = sx.classify_stagex(good_records(), baseline(), selections(), final_policy(3), sx.StageXSpec()); assert result['primary_failure_locus'] == 'outer_policy_instability'


def test_classification_metric_failure():
    records = good_records(); records[25] = metric(overall=1.01)
    result = sx.classify_stagex(records, baseline(), selections(), final_policy(), sx.StageXSpec()); assert result['scientific_status'] == 'BLOCKED'


def stagew_report():
    return {
        'execution_verdict':'PASS','scientific_status':'BLOCKED',
        'root_cause':'phase314b_r258_stagew_aggregate_evidence_localizes_failure_to_accepted_row_fidelity_with_minority_adverse_sse_dominance',
        'required_next_path':'DESIGN_OBJECTIVE_TRAIN_ONLY_TAIL_ROBUST_DIRECTION_CALIBRATION_WITH_NESTED_GROUP_OOF_WHILE_KEEPING_FROZEN_PROBE_CLOSED',
        'selected_configuration':None,'train_only_recommendation':None,
        'cumulative_selection_holdout_evaluation_count':1,'rerun_authorized':False,
    }


def test_stagew_report_validates():
    sx.validate_stagew_report(stagew_report())


def test_stagew_report_rejects_rerun():
    value = stagew_report(); value['rerun_authorized'] = True
    with pytest.raises(sx.StageXError): sx.validate_stagew_report(value)


def test_write_once(tmp_path):
    path = tmp_path/'x.json'; sx.write_once(path, b'{}\n'); assert path.read_bytes() == b'{}\n'
    with pytest.raises(sx.StageXError): sx.write_once(path, b'{}\n')


def worker_payload(status='BLOCKED'):
    payload = {
        'schema':sx.WORKER_SCHEMA,'execution_verdict':'PASS','scientific_status':status,
        'selection_holdout_evaluation_count_added':0,'cumulative_selection_holdout_evaluation_count':1,
        'selected_configuration':None,'rerun_authorized':False,
        'tail_robust_nested_oof_records':{'10':{},'25':{},'50':{}},
        'train_only_recommendation': None if status == 'BLOCKED' else {'x':1},
    }
    payload.update({key:False for key in sx.FALSE_BOUNDARIES})
    payload['scientific_result_sha256'] = sx.sha256_bytes(sx.stable_json_bytes(payload))
    return payload


def test_worker_payload_validates_blocked():
    sx.validate_worker_payload(worker_payload())


def test_worker_payload_validates_ready():
    sx.validate_worker_payload(worker_payload('READY'))


def test_worker_payload_rejects_boundary():
    value = worker_payload(); value[sx.FALSE_BOUNDARIES[0]] = True
    value['scientific_result_sha256'] = sx.sha256_bytes(sx.stable_json_bytes({k:v for k,v in value.items() if k!='scientific_result_sha256'}))
    with pytest.raises(sx.StageXError): sx.validate_worker_payload(value)


def test_worker_payload_rejects_hash():
    value = worker_payload(); value['scientific_result_sha256'] = '0'*64
    with pytest.raises(sx.StageXError): sx.validate_worker_payload(value)


def test_blocked_report_preserves_boundaries():
    value = sx.blocked_report(None, RuntimeError('x')); assert value['rerun_authorized'] is False; assert value['selection_holdout_evaluation_count_added'] == 0
