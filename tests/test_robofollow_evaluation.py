import json
import pytest

def output(tmp_path,status='complete',completed=2):
    (tmp_path/'run.json').write_text(json.dumps(dict(status=status,expected_trials=2,completed_trials=completed,selected_tasks=[{'scene':'scene1','task':'a'}],arguments={'rounds':2},metric_version='strict',completion_aggregation='task-macro')))
    rows=[dict(scene='scene1',level_tag='L0',task='a',round=i,seed=i,intent_score=.5,exec_score=.75,intent_ok=False,exec_ok=False,completion_ok=i==1) for i in range(1,3)]
    (tmp_path/'results.json').write_text(json.dumps(rows))
    (tmp_path/'protocol.json').write_text(json.dumps(dict(hash='fixed',expected_trials=2,selected_tasks=[{'scene':'scene1','task':'a'}],arguments={'rounds':2})))
    return tmp_path

def test_complete_metrics_keep_distinct_scores(tmp_path):
    from autoresearch.evaluation import read_complete_metrics
    m=read_complete_metrics(output(tmp_path),'fixed')
    assert m['mean_intent_score']==.5 and m['mean_exec_score']==.75 and m['completion_rate']==.5
    assert 'scene1/L0' in m['groups']

@pytest.mark.parametrize('status,completed',[('running',1),('complete',1),('complete',0)])
def test_partial_rejected(tmp_path,status,completed):
    from autoresearch.evaluation import read_complete_metrics
    with pytest.raises(ValueError):read_complete_metrics(output(tmp_path,status,completed),'fixed')

def test_nan_protocol_and_empty_rejected(tmp_path):
    from autoresearch.evaluation import read_complete_metrics
    output(tmp_path)
    with pytest.raises(ValueError):read_complete_metrics(tmp_path,'other')
    rows=json.loads((tmp_path/'results.json').read_text());rows[0]['intent_score']=float('nan');(tmp_path/'results.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError):read_complete_metrics(tmp_path,'fixed')
    (tmp_path/'results.json').write_text('[]')
    with pytest.raises(ValueError):read_complete_metrics(tmp_path,'fixed')

def test_intent_gain_with_execution_regression_rejected():
    from autoresearch.evaluation import compare_candidate
    b=dict(protocol_hash='fixed',mean_intent_score=.2,mean_exec_score=.7,completion_rate=.6)
    c=dict(b,mean_intent_score=.4,mean_exec_score=.4)
    t=dict(frozen=True,min_intent_gain=.01,exec_margin=.03,completion_margin=.03)
    assert compare_candidate(b,c,t)['decision']=='reject'
    c['mean_exec_score']=.8
    assert compare_candidate(b,c,t)['decision']=='keep'
    with pytest.raises(ValueError):compare_candidate(b,c,{})

def test_forged_actual_arguments_and_duplicate_round_rejected(tmp_path):
    from autoresearch.evaluation import read_complete_metrics
    output(tmp_path)
    run=json.loads((tmp_path/'run.json').read_text());run['arguments']['rounds']=3;(tmp_path/'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError):read_complete_metrics(tmp_path,'fixed')
    output(tmp_path)
    rows=json.loads((tmp_path/'results.json').read_text());rows[1]['round']=1;(tmp_path/'results.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError):read_complete_metrics(tmp_path,'fixed')

def test_calibration_freezes_before_comparison():
    from autoresearch.evaluation import calibrate_thresholds
    rows=[dict(scene='scene1',task=t,intent_score=.5,exec_score=.75,completion_ok=True) for t in ('a','b') for _ in range(2)]
    m=calibrate_thresholds(rows)
    assert m['frozen'] and m['min_intent_gain']>0 and m['calibration_tasks']==2
    with pytest.raises(ValueError):calibrate_thresholds(rows[:1])

def test_official_one_based_rounds_accepted(tmp_path):
    from autoresearch.evaluation import read_complete_metrics
    output(tmp_path)
    rows=json.loads((tmp_path/'results.json').read_text())
    (tmp_path/'results.json').write_text(json.dumps(rows))
    assert read_complete_metrics(tmp_path,'fixed')['episodes']==2
