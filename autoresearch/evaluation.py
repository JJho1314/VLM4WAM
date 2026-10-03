"""Admission uses complete, protocol-matched official RoboFollow results."""
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

def paired_language_uncertainty(correct,shuffled):
    identity=lambda r:(r['scene'],r['task'],r['round'],r['seed'])
    if [identity(r) for r in correct]!=[identity(r) for r in shuffled]:raise ValueError('language trials are not paired')
    tasks=defaultdict(list)
    for a,b in zip(correct,shuffled):
        delta=a['intent_score']-b['intent_score']
        if not math.isfinite(delta):raise ValueError('nonfinite paired metric')
        tasks[(a['scene'],a['task'])].append(delta)
    if len(tasks)<2:raise ValueError('language uncertainty needs multiple tasks')
    values=[mean(v) for v in tasks.values()];average=mean(values);se=stdev(values)/math.sqrt(len(values))
    return dict(mean=average,standard_error=se,ci95=[max(-1.,average-1.96*se),min(1.,average+1.96*se)],tasks=len(tasks),episodes=len(correct),method='approximate normal interval over task means; small fixed development subset, no benchmark generalization')


def read_complete_metrics(directory, protocol_hash):
    p=Path(directory)
    run=json.loads((p/'run.json').read_text())
    protocol=json.loads((p/'protocol.json').read_text())
    rows=json.loads((p/'results.json').read_text())
    expected=protocol['expected_trials']
    if (protocol['hash']!=protocol_hash or run['status']!='complete' or expected<=0
        or run['expected_trials']!=expected or run['completed_trials']!=expected or len(rows)!=expected
        or run['selected_tasks']!=protocol['selected_tasks']
        or any(run['arguments'].get(k)!=v for k,v in protocol['arguments'].items())
        or run.get('metric_version')!='strict' or run.get('completion_aggregation')!='task-macro'):
        raise ValueError('incomplete or mismatched evaluation protocol')
    if protocol.get('official_code_sha256') and run.get('code_sha256')!=protocol['official_code_sha256']:raise ValueError('official code changed')
    selected={(x['scene'],x['task']) for x in protocol['selected_tasks']}
    identities=set();groups=defaultdict(list);tasks=defaultdict(list)
    for r in rows:
        ident=(r['scene'],r['task'],r['round'])
        if ident in identities or ident[:2] not in selected:raise ValueError('duplicate or unexpected trial')
        identities.add(ident)
        for k in ('intent_score','exec_score'):
            if not isinstance(r[k],(float,int)) or not math.isfinite(r[k]) or not 0<=r[k]<=1:raise ValueError('invalid metric')
        for k in ('intent_ok','exec_ok','completion_ok'):
            if not isinstance(r[k],bool):raise ValueError('invalid success flag')
        groups[r['scene']+'/'+r['level_tag']].append(r)
        tasks[ident[:2]].append(r)
    rounds=run['arguments']['rounds']
    if set(tasks)!=selected or any({r['round'] for r in rs}!=set(range(1,rounds+1)) for rs in tasks.values()):raise ValueError('missing rounds')
    def summary(rs):
        ts=defaultdict(list)
        for r in rs:ts[(r['scene'],r['task'])].append(r['completion_ok'])
        return dict(mean_intent_score=mean(r['intent_score'] for r in rs),mean_exec_score=mean(r['exec_score'] for r in rs),completion_rate=mean(mean(v) for v in ts.values()),intent_full_rate=mean(r['intent_ok'] for r in rs),exec_full_rate=mean(r['exec_ok'] for r in rs),episode_micro_completion_rate=mean(r['completion_ok'] for r in rs),episodes=len(rs))
    return dict(summary(rows),protocol_hash=protocol_hash,groups={k:summary(v) for k,v in groups.items()})


def calibrate_thresholds(rows,output=None):
    """Freeze uncertainty margins from baseline repeats before candidate evaluation."""
    tasks=defaultdict(list)
    for r in rows:tasks[(r['scene'],r['task'])].append(r)
    if len(tasks)<2 or any(len(v)<2 for v in tasks.values()):raise ValueError('calibration needs at least two tasks and two rounds')
    def margin(key):
        values=[mean(r[key] for r in rs) for rs in tasks.values()]
        if any(not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('invalid calibration metric')
        return 1.96*stdev(values)/math.sqrt(len(values))
    result=dict(version=1,frozen=True,min_intent_gain=max(.01,margin('intent_score')),exec_margin=max(.02,margin('exec_score')),completion_margin=max(.02,margin('completion_ok')),calibration_tasks=len(tasks))
    import hashlib
    result['baseline_trials_sha256']=hashlib.sha256(json.dumps(rows,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if output is not None:
        from autoresearch.registry import atomic_json
        atomic_json(output,result)
    return result


def compare_candidate(baseline,candidate,thresholds):
    keys=('min_intent_gain','exec_margin','completion_margin')
    if thresholds.get('frozen') is not True or any(k not in thresholds or not math.isfinite(thresholds[k]) or thresholds[k]<0 for k in keys):raise ValueError('frozen calibrated thresholds required')
    if baseline['protocol_hash']!=candidate['protocol_hash']:raise ValueError('protocol mismatch')
    if baseline.get('runtime_fingerprint')!=candidate.get('runtime_fingerprint'):raise ValueError('runtime or asset fingerprint mismatch')
    deltas={k:candidate[k]-baseline[k] for k in ('mean_intent_score','mean_exec_score','completion_rate')}
    if any(not math.isfinite(v) for v in deltas.values()):raise ValueError('nonfinite metrics')
    if deltas['mean_exec_score'] < -thresholds['exec_margin'] or deltas['completion_rate'] < -thresholds['completion_margin'] or deltas['mean_intent_score']<0:
        decision='reject'
    elif deltas['mean_intent_score']>=thresholds['min_intent_gain']:decision='keep'
    else:decision='inconclusive'
    return dict(decision=decision,deltas=deltas)
