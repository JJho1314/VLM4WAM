"""Registered paired evaluation worker; its parent owns a bounded process group."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from autoresearch.registry import atomic_json,stop_process_tree
from autoresearch.evaluation import read_complete_metrics

PIX='/data/users/junjie/.pixi/bin/pixi'
ROOT=Path(__file__).resolve().parents[1]


def verify_protocol(protocol):
    data=dict(protocol);digest=data.pop('hash')
    if hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=digest:raise ValueError('protocol digest mismatch')
    if protocol['instruction_modes']!=['correct','shuffle'] or protocol['metric_version']!='strict' or protocol['completion_aggregation']!='task-macro':raise ValueError('unregistered protocol')
    if len(protocol['selected_tasks'])!=4 or {x['scene'] for x in protocol['selected_tasks']}!={'scene1','scene2','scene3','scene4'}:raise ValueError('protocol requires one task per scene')
    return protocol


def wait_service(proc,host,port,timeout=600):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if proc.poll() is not None:raise RuntimeError('policy service failed during startup')
        try:
            with socket.create_connection((host,port),timeout=.2):return
        except OSError:time.sleep(.5)
    raise TimeoutError('policy service readiness timeout')


def evaluate_pair(directory,config_path,checkpoint_path,protocol_path,planner_mode):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    protocol=verify_protocol(json.loads(Path(protocol_path).read_text()))
    config_path=Path(config_path).resolve()
    import yaml
    config=yaml.safe_load(config_path.read_text());data=config['data']['train']
    manifest=Path(data['manifest_path'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=protocol['manifest_sha256']:raise ValueError('development data protocol mismatch')
    # Refuse an occupied port, avoiding requests to someone else's server.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',8999))
    pixi=[PIX,'run','--manifest-path',str(ROOT/'autoresearch/runtime/pixi.toml')]
    for mode in protocol['instruction_modes']:
        kwargs=dict(config_path=str(config_path),checkpoint_path=str(Path(checkpoint_path).resolve()),stats_path=data['stat_file'],manifest_path=str(manifest),planner_mode=planner_mode,instruction_mode=mode)
        argv=pixi+['serve','--factory','ge_act.experiments.robofollow_policy:make_policy','--kwargs',json.dumps(kwargs),'--host','127.0.0.1','--port','8999']
        with (directory/f'{mode}_serve.log').open('w') as log:
            server=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),stdout=log,stderr=subprocess.STDOUT)
            try:
                wait_service(server,'127.0.0.1',8999)
                for task in protocol['selected_tasks']:
                    out=directory/mode/task['scene']
                    # Every attempt is new; official evaluator itself cannot resume.
                    if out.exists():raise FileExistsError(out)
                    args=protocol['arguments']
                    cmd=pixi+['sim','-m','robofollow.evaluate','--scene',task['scene'],'--levels','L0','--tasks',task['task'],'--rounds',str(args['rounds']),'--base-seed',str(args['base_seed']),'--max-steps',str(args['max_steps']),'--actions-per-step',str(args['actions_per_step']),'--runtime',args['runtime'],'--sim-steps',str(args['sim_steps']),'--gpu','1','--remote','--host','127.0.0.1','--port','8999','--output',str(out)]
                    atomic_json(directory/f'{mode}_{task["scene"]}_argv.json',cmd)
                    with (directory/f'{mode}_{task["scene"]}.log').open('w') as eval_log:subprocess.run(cmd,cwd=ROOT,stdout=eval_log,stderr=subprocess.STDOUT,check=True)
                    sidecar=dict(hash=protocol['hash'],expected_trials=args['rounds'],selected_tasks=[task],arguments=args,official_code_sha256=protocol['official_code_sha256'])
                    atomic_json(out/'protocol.json',sidecar)
                    read_complete_metrics(out,protocol['hash'])
            finally:
                stop_process_tree(server)
    atomic_json(directory/'complete.json',dict(status='complete',protocol_hash=protocol['hash']))


def collect_pair(directory,protocol):
    from statistics import mean
    directory=Path(directory);verify_protocol(protocol)
    complete=json.loads((directory/'complete.json').read_text())
    if complete!=dict(status='complete',protocol_hash=protocol['hash']):raise ValueError('paired evaluation incomplete')
    output={};raw={};runtime_hashes=set()
    for mode in protocol['instruction_modes']:
        groups={};rows=[]
        for task in protocol['selected_tasks']:
            p=directory/mode/task['scene'];metrics=read_complete_metrics(p,protocol['hash'])
            actual=json.loads((p/'run.json').read_text())
            runtime={k:actual[k] for k in ('gate_mode','completion_metric_version','stepper','versions','curobo_source_sha256','config_sha256','asset_overrides','code_sha256')}
            runtime_hashes.add(hashlib.sha256(json.dumps(runtime,sort_keys=True,separators=(',',':')).encode()).hexdigest())
            if actual['selected_tasks']!=[task] or actual.get('code_sha256')!=protocol['official_code_sha256']:raise ValueError('paired task/code mismatch')
            groups.update(metrics['groups']);rows.extend(json.loads((p/'results.json').read_text()))
        output[mode]=dict(protocol_hash=protocol['hash'],mean_intent_score=mean(r['intent_score'] for r in rows),mean_exec_score=mean(r['exec_score'] for r in rows),completion_rate=mean(v['completion_rate'] for v in groups.values()),groups=groups,episodes=len(rows))
        raw[mode]=rows
    if len(runtime_hashes)!=1:raise ValueError('runtime or assets changed between paired trials')
    for mode in protocol['instruction_modes']:output[mode]['runtime_fingerprint']=next(iter(runtime_hashes))
    identities=lambda rows:[(r['scene'],r['task'],r['round'],r['seed']) for r in rows]
    if identities(raw['correct'])!=identities(raw['shuffle']):raise ValueError('language ablation trials are not paired')
    output['language_delta']={k:output['correct'][k]-output['shuffle'][k] for k in ('mean_intent_score','mean_exec_score','completion_rate')}
    from autoresearch.evaluation import paired_language_uncertainty
    output['language_uncertainty']=paired_language_uncertainty(raw['correct'],raw['shuffle'])
    return output,raw


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--protocol',required=True);p.add_argument('--planner-mode',choices=['disabled','predicted'],default='disabled');a=p.parse_args()
    evaluate_pair(a.directory,a.config,a.checkpoint,a.protocol,a.planner_mode)
    result,_=collect_pair(a.directory,json.loads(Path(a.protocol).read_text()));atomic_json(Path(a.directory)/'metrics.json',result)

if __name__=='__main__':main()
