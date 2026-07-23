from __future__ import annotations
import inspect, json
from pathlib import Path
import pytest
from ccda_phase3 import phase314b_r258_stagev_locked_selection_holdout_evaluation as v


def real_contract() -> dict:
    path=Path(__file__).resolve().parents[1]/v.BASE_CONTRACT
    assert path.is_file(), 'real Stage-U contract must be present'
    return json.loads(path.read_text(encoding='utf-8'))


def metric(overall=.9, accepted=.9, positive=.6, relative=.1, mechanism=True):
    return {'mechanism_eligible':mechanism,'overall_mse_ratio':overall,'accepted_rows':{'mse_ratio':accepted,'positive_distance_reduction_rate':positive,'relative_distance_reduction':{'mean':relative}}}

def candidate(mismatch=0, aligned=0, regression=0):
    return {'length_log_z_element_mismatch_count':mismatch,'aligned_upper_element_failure_count':aligned,'strict_pass_aligned_fail_row_count':regression}

def record(t, passed=True, checks=None):
    c=checks or {'mechanism_eligible':passed,'overall_mse_ratio':passed,'accepted_row_mse_ratio':passed,'positive_distance_reduction_rate':passed,'relative_distance_reduction_mean':passed,'length_log_z_element_mismatch_count':passed,'aligned_upper_element_failure_count':passed,'strict_pass_aligned_fail_row_count':passed,'all':passed}
    return {'timestep':t,'scientific_pass':passed,'pass_checks':c}


def test_real_stageu_contract_schema_and_hash():
    validated=v.validate_stageu_contract(real_contract()); assert validated['frontier']['backbone_id']==v.LOCKED_BACKBONE

def test_reconstruct_has_no_target_parameter():
    assert 'target' not in inspect.signature(v.reconstruct_locked_holdout_candidate).parameters

def test_guard_blocks_and_then_counts_once():
    g=v.HoldoutTargetGuard({'holdout_target':123,'x':1}); assert g['x']==1
    with pytest.raises(v.StageVError): _=g['holdout_target']
    g.unlock(); assert g['holdout_target']==123; assert g.access_count==1

def test_guard_rejects_second_unlock():
    g=v.HoldoutTargetGuard({}); g.unlock()
    with pytest.raises(v.StageVError): g.unlock()

@pytest.mark.parametrize('field,value',[
('overall_mse_ratio',1.0),('accepted_mse_ratio',1.0),('positive_rate',.5),('relative_mean',0.0)])
def test_strict_fidelity_boundaries_fail(field,value):
    kwargs={'overall':.9,'accepted':.9,'positive':.6,'relative':.1}
    kwargs[{'overall_mse_ratio':'overall','accepted_mse_ratio':'accepted','positive_rate':'positive','relative_mean':'relative'}[field]]=value
    assert v.per_timestep_checks(metric(**kwargs),candidate())['all'] is False

@pytest.mark.parametrize('field', ['length_log_z_element_mismatch_count','aligned_upper_element_failure_count','strict_pass_aligned_fail_row_count'])
def test_any_gate_count_fails(field):
    c=candidate(); c[field]=1; assert v.per_timestep_checks(metric(),c)['all'] is False

def test_all_checks_pass(): assert v.per_timestep_checks(metric(),candidate())['all'] is True

def test_classify_all_pass_ready():
    result=v.classify([record(10),record(25),record(50)]); assert result['scientific_status']=='READY'; assert result['failed_timesteps']==[]

def test_classify_any_fail_blocks_entire_frontier():
    result=v.classify([record(10),record(25,False),record(50)]); assert result['scientific_status']=='BLOCKED'; assert result['failed_timesteps']==[25]

def test_classify_gate_regression():
    checks=record(25,False)['pass_checks']; checks['length_log_z_element_mismatch_count']=False
    result=v.classify([record(10),record(25,False,checks),record(50)]); assert result['primary_failure_locus']=='mechanism_gate_regression'

def test_blocked_report_claims_no_totals_and_no_rerun():
    x=v.blocked_report(repository=None,error=RuntimeError('x')); assert x['complete_fit_count_claimed'] is False; assert x['rerun_authorized'] is False; assert x['selected_configuration'] is None

def test_output_paths_distinct(): assert v.SUCCESS_REPORT != v.BLOCKED_REPORT

def test_locked_policy_constants(): assert v.LOCKED_TIMESTEPS==(10,25,50) and v.LOCKED_BACKBONE=='segment_target_rr64_feasible'

def test_single_worker_source_contract():
    text=(Path(__file__).resolve().parents[1]/'ccda_phase3/phase314b_r258_stagev_locked_selection_holdout_evaluation.py').read_text()
    assert '"cold_science_worker_count": 1' in text
    assert 'fallback_backbone_used": False' in text
    assert 'fallback_timestep_used": False' in text

def test_worker_script_does_not_probe_environment():
    text=(Path(__file__).resolve().parents[1]/'scripts/phase3_14b_r258_stagev_worker.py').read_text(); assert 'environment-probe' not in text

def test_execute_has_write_once_precheck():
    text=(Path(__file__).resolve().parents[1]/'scripts/phase3_14b_r258_stagev_execute.py').read_text(); assert 'rerun is forbidden' in text
