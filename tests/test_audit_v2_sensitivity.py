"""CPU checks for response-consistent partial sensitivity and crossed bootstrap."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('sensitivity_v2_test',Path(__file__).parents[1]/'analysis/audit_v2_sensitivity.py')
s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)


def unit(text='Paris',truncated=False,automatic='error'):
    return {'response':text,'truncated':truncated,'automatic_outcome':automatic}


@pytest.mark.parametrize('u,h,expected,reason',[
    (unit(),'correct',1,'audited_semantic_label'),
    (unit(),'error',0,'audited_semantic_label'),
    (unit(),None,0,'unaudited_em_retained'),
    (unit('IDK',False,'abstain'),'correct',-1,'fixed_literal_idk'),
    (unit('Paris',True),'correct',0,'fixed_invalid_output'),
    (unit(''),'correct',0,'fixed_invalid_output'),
    (unit('I do not know'),'abstain',0,'semantic_abstention_strict_format_error'),
    (unit('.IDK'),'abstain',0,'semantic_abstention_strict_format_error')])
def test_strict_format_contract_and_unaudited_em(u,h,expected,reason):
    assert s.strict_semantic_code(u,h)==(expected,reason)


def test_unclear_remains_distinct_from_hypothetical_assignments():
    u=unit()
    assert s.strict_semantic_code(u,'unclear')==(0,'unclear_em_retained')
    assert s.strict_semantic_code(u,'unclear',1)==(1,'hypothetical_unclear_assignment')
    assert s.strict_semantic_code(u,'unclear',0)==(0,'hypothetical_unclear_assignment')
    with pytest.raises(ValueError):s.strict_semantic_code(u,'unclear',-1)


def test_unit_assignment_propagates_to_all_occurrences_but_not_rejected_filters():
    original=np.array([[[0,0],[-1,0]],[[0,0],[-1,0]],[[0,0],[-1,0]]],dtype=np.int8)
    known=original.copy();known[:,1,1]=1
    masks=np.zeros((1,*original.shape),bool);masks[0,:,0,:]=True
    comps,base=s.assemble_components(original,known,masks,[0])
    assert comps.shape==(3,3,2)
    assert np.array_equal(s.scenario_codes(base,masks,[0]),known)
    hypothetical=s.scenario_codes(base,masks,[1])
    assert np.all(hypothetical[:,0,:]==1) and np.all(hypothetical[:,1,0]==-1)
    assert np.array_equal(hypothetical>=0,original>=0)
    point=comps.mean(axis=(1,2))
    assert s.scenario_value(point,[1])==pytest.approx(s.utility(hypothetical).mean())
    assert s.scenario_value(point,[0])==pytest.approx(s.utility(known).mean())


def test_coherence_can_cancel_unclear_unit_in_paired_contrast():
    left=np.array([-.02,.03,.1,.04]);right=np.array([-.03,.02,.1,0])
    # First unresolved response affects both methods equally: cancels exactly.
    delta=left-right
    assert s.scenario_value(delta,[0,0])==pytest.approx(s.scenario_value(delta,[1,0]))
    assert s.scenario_value(delta,[1,1])-s.scenario_value(delta,[0,0])==pytest.approx(.04)


def test_masks_cannot_overlap_or_modify_emission_decisions():
    o=np.zeros((3,2,2),dtype=np.int8);m=np.ones((2,3,2,2),bool)
    with pytest.raises(ValueError,match='overlap'):s.assemble_components(o,o,m,[0,0])
    k=o.copy();k[0,0,0]=-1
    with pytest.raises(ValueError,match='emission'):s.assemble_components(o,k,np.zeros((0,3,2,2),bool),[])


def test_crossed_components_bootstrap_matches_direct_sampling_and_deterministic_baseline():
    values=np.arange(2*4*3*5,dtype=float).reshape(2,4,3,5)/100
    values[1]=np.repeat(values[1,:,:1,:],3,axis=1)
    result=s.bootstrap_components(values,77,123)
    rng=np.random.default_rng(123);expected=[]
    for _ in range(77):
        ss=rng.integers(0,3,3);qs=rng.integers(0,5,5)
        expected.append(values[:,:,ss][:,:,:,qs].mean(axis=(2,3)))
    assert np.allclose(result,expected,rtol=0,atol=1e-14)
    assert np.allclose(result[:,0,1]-result[:,1,1],np.asarray(expected)[:,0,1]-np.asarray(expected)[:,1,1])
    with pytest.raises(ValueError):s.bootstrap_components(values[:,:,:1,:],10)


def test_interval_envelopes_distinct_from_identification_bounds():
    point=np.array([-.1,.02,.03,-.01])
    boot=np.tile(point,(100,1));boot[:,1]+=np.linspace(-.02,.02,100)
    assignments=[(0,0),(0,1),(1,0),(1,1)]
    summary,cases=s.scenario_summary(point,boot,assignments,(0,0))
    assert summary['coherent_point_low']==pytest.approx(.01)
    assert summary['coherent_point_high']==pytest.approx(.05)
    assert summary['scenario_ci95_low_min'] < summary['coherent_point_low']
    assert summary['scenario_ci95_high_max'] > summary['coherent_point_high']
    assert summary['all_assignments_ci95_positive'] is False
    assert summary['partial_em_unclear_point']==pytest.approx(.02)
    assert len(cases)==4


def test_risk_zero_coverage_and_gain_error_to_correct_one_unit():
    blank=s.summarized(np.full((3,5,2),-1))
    assert blank['selective_risk'] is None and blank['coverage']==0 and blank['utility']==0
    original=np.array([[[0,0]]]);correct=np.array([[[1,1]]])
    assert s.summarized(correct)['utility']-s.summarized(original)['utility']==pytest.approx(1)


def test_output_refuses_overwrite_before_reading(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'load_inputs',lambda *a:pytest.fail('should not read'))
    with pytest.raises(FileExistsError):s.analyze(tmp_path)
