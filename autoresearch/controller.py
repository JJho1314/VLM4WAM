"""Bounded runner. Candidate commands are constructed from registered templates."""
from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from autoresearch.registry import Registry,atomic_json,process_identity,safe_signal,descendant_identities,cleanup_owned_group,owned_group_identities

PIX='/data/users/junjie/.pixi/bin/pixi'
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=Path('/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_autoresearch/runs')

def candidate_config(variant,data_version='frozen-v1'):
    configs={'text':'action_model_text.yaml','predicted':'action_model_planner.yaml','mix':'action_model_mix.yaml','joint':'action_model_joint.yaml'}
    if variant not in configs:raise ValueError('unregistered variant')
    if data_version not in ('frozen-v1','complete-v2','complete-v3'):raise ValueError('unregistered data version')
    return configs[variant] if data_version=='frozen-v1' else configs[variant].replace('.yaml','_'+data_version.replace('-','_')+'.yaml')

def require_admission(variant,root,marker=None,protocol_hash=None):
    if variant=='text':return
    if not all((Path(root)/x).exists() for x in ('baseline.json','thresholds.json')):
        raise ValueError('complete and freeze baseline before preparing planner candidates')
    if protocol_hash is not None and json.loads((Path(root)/'thresholds.json').read_text()).get('protocol_hash')!=protocol_hash:
        raise ValueError('baseline protocol and data version mismatch')
    marker=Path(marker or ROOT/'autoresearch/configs/planner_validation.json')
    if not marker.exists() or variant not in json.loads(marker.read_text()).get('validated_variants',[]):
        raise ValueError('planner candidate lacks validated GPU integration')

def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def launch_environment(**overrides):
    # Registered YAML is authoritative; inherited experimental knobs are forbidden.
    env={k:v for k,v in os.environ.items() if not k.startswith('BATON_RESEARCH_')}
    env['BATON_RESEARCH_CAPTION_DROPOUT']='0'
    env.update(overrides)
    return env

def refuse_unresolved(registry):
    pending=[str(p) for p in registry.root.glob('*/*/process.json') if json.loads(p.read_text()).get('status')=='running']
    if pending:raise RuntimeError(f'unresolved GPU process records; recover before launch: {pending}')

@contextmanager
def termination_cleanup():
    def handler(signum,frame):raise KeyboardInterrupt(f'controller signal {signum}')
    previous={sig:signal.signal(sig,handler) for sig in (signal.SIGTERM,signal.SIGHUP)}
    try:yield
    finally:
        for sig,value in previous.items():signal.signal(sig,value)

def require_image_validation(manifest):
    manifest=Path(manifest);data=json.loads(manifest.read_text())
    if data.get('image_validation')=='all usable frames, streaming RGB decode':return
    receipt=manifest.parent/'all_frames_validation.json'
    result=json.loads(receipt.read_text()) if receipt.exists() else {}
    if result.get('status')!='complete' or result.get('manifest_sha256')!=file_sha256(manifest) or result.get('episodes')!=len(data['episodes']):
        raise ValueError('complete image validation required before GPU admission')

def bind_run_checkpoint(run,checkpoint):
    import yaml
    from ge_act.experiments.robofollow_loading import semantic_contract,write_contract
    state=json.loads((run/'run.json').read_text())
    config_path=run/'code/ge_act/configs/ltx_model/robofollow/run.yaml'
    if state.get('training',{}).get('status')!='completed':raise ValueError('contract requires completed registered training')
    if file_sha256(config_path)!=state['effective_config_sha256']:raise ValueError('effective training config changed')
    expected=run/'checkpoints'
    if expected.resolve() not in checkpoint.resolve().parents:raise ValueError('checkpoint outside registered run')
    source=json.loads((run/'provenance.json').read_text())
    for path,info in source['artifacts'].items():
        if file_sha256(path)!=info['sha256']:raise ValueError('training artifact changed since preparation: '+path)
    config=yaml.safe_load(config_path.read_text())
    require_image_validation(config['data']['train']['manifest_path'])
    write_contract(checkpoint,semantic_contract(config,config['data']['train']['stat_file']),dict(capture_phase='post-training audited binding',run_id=run.name,effective_config_sha256=state['effective_config_sha256'],training_process=state['training']['process']))

def provenance(config):
    import importlib.metadata
    paths=[config['diffusion_model']['model_path'],config['data']['train']['manifest_path'],config['data']['train']['stat_file']]
    planner=config.get('robofollow_planner',{})
    if planner.get('enabled'):
        for label in ('checkpoint','qwen_path','siglip_path'):
            base=Path(planner[label])
            paths.extend([str(base)] if base.is_file() else (str(p) for p in base.rglob('*') if p.is_file()))
    artifacts={str(p):dict(sha256=file_sha256(p),bytes=Path(p).stat().st_size) for p in paths}
    effective=launch_environment()
    env={key:effective[key] for key in ('CUDA_VISIBLE_DEVICES','HDF5_USE_FILE_LOCKING','PYTHONPATH','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','VK_ICD_FILENAMES','CUDA_HOME','BATON_RESEARCH_CAPTION_DROPOUT') if key in effective}
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    return dict(captured_at=time.time(),artifacts=artifacts,environment=env,packages=packages,external_models=dict(pretrained=config['pretrained_model_name_or_path'],qwen=planner.get('qwen_path'),siglip=planner.get('siglip_path')),runtime_note='See captured Pixi manifest for separate simulator interpreter and environment')


def validate_candidate(c):
    if set(c)-{'id','variant','steps','timeout_seconds','data_version'} or c.get('id') not in ('R0','R1','R2','R3') or c.get('variant') not in ('text','predicted','mix','joint'):
        raise ValueError('unregistered candidate')
    if c.get('data_version','frozen-v1') not in ('frozen-v1','complete-v2','complete-v3'):raise ValueError('unregistered data version')
    if {'R0':'text','R1':'predicted','R2':'mix','R3':'joint'}[c['id']]!=c['variant']:raise ValueError('candidate variant mismatch')
    if not isinstance(c.get('steps'),int) or not 1<=c['steps']<=300 or not 0<c.get('timeout_seconds',0)<=7200:raise ValueError('candidate exceeds limits')


def snapshot_code(src,dst):
    src=Path(src);dst=Path(dst);dst.mkdir(parents=True,exist_ok=False)
    def git(*args):return subprocess.check_output(['git','-C',str(src),*args])
    head=git('rev-parse','HEAD').decode().strip()
    files=set(git('ls-files','-z').split(b'\0'))|set(git('ls-files','--others','--exclude-standard','-z').split(b'\0'))
    expanded=set()
    for raw in files:
        if not raw:continue
        p=src/os.fsdecode(raw)
        if p.is_dir():
            expanded.update(os.fsencode(str(x.relative_to(src))) for x in p.rglob('*') if x.is_file() and '.git' not in x.relative_to(p).parts and '__pycache__' not in x.parts)
        else:expanded.add(raw)
    excluded=[];hashes={}
    for raw in sorted(expanded):
        if not raw:continue
        name=os.fsdecode(raw);p=src/name
        if not p.exists():continue
        if p.is_symlink():raise ValueError(f'snapshot refuses symlink: {name}')
        if p.stat().st_size>50*1024*1024 or p.suffix in ('.hdf5','.safetensors','.pt','.mp4'):
            excluded.append(name);continue
        target=dst/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    (dst/'.dirty.patch').write_bytes(git('diff','HEAD','--binary'))
    atomic_json(dst/'.snapshot.json',dict(head=head,sha256=hashes,excluded_large_artifacts=excluded))


def run_process(argv,directory,registry,timeout_seconds,gpus,env=None,cwd=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    refuse_unresolved(registry)
    registry.check_budget(timeout_seconds*gpus/3600)
    start=time.time();clock_start=time.monotonic();record=dict(status='running',argv=argv,started_at=start,started_monotonic=clock_start,gpus=gpus,timeout_seconds=timeout_seconds,controller=process_identity(os.getpid()))
    with (directory/'process.log').open('w') as log:
        proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env if env is not None else launch_environment(),cwd=cwd)
        watch=None;watch_identity=None;watch_log=None
        try:
            record['process']=process_identity(proc.pid);record['owned_processes']=descendant_identities(proc.pid);atomic_json(directory/'process.json',record)
            watch_log=(directory/'watchdog.log').open('w')
            watch=subprocess.Popen([PIX,'run','--manifest-path',str(ROOT/'autoresearch/runtime/pixi.toml'),'research','-m','autoresearch.watchdog','--record',str(directory/'process.json'),'--owner',json.dumps(process_identity(os.getpid())),'--root',str(registry.root),'--maximum',str(registry.max_gpu_hours)],start_new_session=True,stdout=watch_log,stderr=subprocess.STDOUT,env=launch_environment(),cwd=ROOT)
            watch_identity=process_identity(watch.pid)
            while proc.poll() is None:
                if process_identity(proc.pid)==record['process']:
                    known={json.dumps(x,sort_keys=True):x for x in record['owned_processes']}
                    known.update({json.dumps(x,sort_keys=True):x for x in descendant_identities(proc.pid)})
                    record['owned_processes']=list(known.values());atomic_json(directory/'process.json',record)
                elapsed=time.monotonic()-clock_start
                registry.charge(str(directory),elapsed,gpus)
                if elapsed>=timeout_seconds or registry.used_gpu_hours()>=registry.max_gpu_hours:
                    record['status']='interrupted';safe_signal(record['process'],signal.SIGTERM,group=True)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:safe_signal(record['process'],signal.SIGKILL,group=True);proc.wait()
                    break
                time.sleep(min(1.,max(.01,timeout_seconds-(time.monotonic()-clock_start))))
            else:record['status']='completed' if proc.returncode==0 else 'failed'
        except BaseException:
            record['status']='interrupted';safe_signal(record['process'],signal.SIGTERM,group=True)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:safe_signal(record['process'],signal.SIGKILL,group=True);proc.wait()
            raise
        finally:
            if watch is not None:
                safe_signal(watch_identity,signal.SIGTERM,group=True)
                try:watch.wait(timeout=10)
                except subprocess.TimeoutExpired:safe_signal(watch_identity,signal.SIGKILL,group=True);watch.wait()
            if watch_log is not None:watch_log.close()
            record['owned_processes']+=owned_group_identities(record['process'])
            cleanup_owned_group(proc.pid,record['owned_processes'])
            record.update(returncode=proc.returncode,ended_at=time.time(),wall_seconds=time.monotonic()-clock_start)
            record['gpu_hours']=record['wall_seconds']*gpus/3600
            registry.charge(str(directory),record['wall_seconds'],gpus);atomic_json(directory/'process.json',record)
    return record


def prepare_run(candidate,root):
    validate_candidate(candidate)
    version=candidate.get('data_version','complete-v3')
    if version!='complete-v3':raise ValueError('superseded split: new runs require complete-v3')
    protocol_name='robofollow_eval_complete_v3.json'
    protocol_hash=json.loads((ROOT/'autoresearch/configs'/protocol_name).read_text())['hash']
    require_admission(candidate['variant'],root,protocol_hash=protocol_hash)
    if candidate['id']=='R0' and (Path(root)/'baseline.json').exists():raise ValueError('baseline already frozen; use a separately budgeted registry for a new data version')
    run=Path(root)/(time.strftime('%Y%m%d_%H%M%S')+'_'+candidate['id']);run.mkdir(parents=True,exist_ok=False)
    snapshot_code(ROOT,run/'code')
    import yaml
    config_path=run/'code/ge_act/configs/ltx_model/robofollow'/candidate_config(candidate['variant'],version)
    config=yaml.safe_load(config_path.read_text())
    require_image_validation(config['data']['train']['manifest_path'])
    atomic_json(run/'provenance.json',dict(provenance(config),config_sha256=file_sha256(config_path)))
    entries=json.loads((ROOT/'autoresearch/configs/robofollow_candidates.json').read_text())
    hypothesis=next(x['hypothesis'] for x in entries if x['id']==candidate['id'])
    atomic_json(run/'run.json',dict(status='prepared',candidate=candidate,hypothesis=hypothesis,protocol_name=protocol_name,created_at=time.time()))
    return run


def elapsed_record(record):
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if 'started_monotonic' in record and record.get('process',{}).get('boot_id')==boot:
        return max(0.,time.monotonic()-record['started_monotonic'])
    return max(0.,time.time()-record['started_at'])

def recover_run(run,registry):
    state=json.loads((run/'run.json').read_text())
    for file in run.glob('*/process.json'):
        record=json.loads(file.read_text())
        if record['status']!='running':continue
        # Live identity is owned by this run; terminate it before billing interruption.
        owner=record.get('process');owned=record.get('owned_processes',[])+owned_group_identities(owner)
        if owner and process_identity(owner['pid'])==owner:owned=owned+descendant_identities(owner['pid'])
        cleanup_owned_group(owner['pid'],owned or [owner]) if owner else None
        elapsed=elapsed_record(record)
        registry.charge(str(file.parent),elapsed,record['gpus'])
        record.update(status='interrupted',ended_at=time.time(),wall_seconds=elapsed,gpu_hours=elapsed*record['gpus']/3600)
        atomic_json(file,record)
    state['status']='interrupted';atomic_json(run/'run.json',state)
    return state



def evaluate_run(run,registry):
    state=json.loads((run/'run.json').read_text())
    if state['status'] not in ('trained','failed','interrupted') or state.get('training',{}).get('status')!='completed':raise ValueError('evaluation requires completed training')
    attempts=state.setdefault('evaluation_attempts',[])
    if 'evaluation' in state and not any(x.get('process')==state['evaluation'].get('process') for x in attempts):attempts.append(state['evaluation'])
    attempt=len(attempts)+1
    while True:
        code_name='evaluation_code' if attempt==1 else f'evaluation_code_attempt_{attempt}'
        phase_name='evaluation' if attempt==1 else f'evaluation_attempt_{attempt}'
        output_name='paired_eval' if attempt==1 else f'paired_eval_attempt_{attempt}'
        if not any((run/name).exists() for name in (code_name,phase_name,output_name)):break
        attempt+=1
    state['active_evaluation_directory']=output_name
    state['active_evaluation_code']=code_name
    checkpoints=list((run/'checkpoints').glob(f"*/step_{state['candidate']['steps']}/diffusion_pytorch_model.safetensors"))
    if len(checkpoints)!=1:raise ValueError('ambiguous or missing final checkpoint')
    remaining=state['candidate']['timeout_seconds']-elapsed_record(state['training'])
    if remaining<=0:raise RuntimeError('candidate total wall-time limit exhausted')
    bind_run_checkpoint(run,checkpoints[0])
    state['checkpoint_sha256']=file_sha256(checkpoints[0])
    snapshot_code(ROOT,run/code_name)
    code=run/code_name
    mode='disabled' if state['candidate']['variant']=='text' else 'predicted'
    argv=[PIX,'run','--manifest-path',str(code/'autoresearch/runtime/pixi.toml'),'research','-m','autoresearch.run_eval','--directory',str(run/output_name),'--config',str(run/'code/ge_act/configs/ltx_model/robofollow/run.yaml'),'--checkpoint',str(checkpoints[0]),'--protocol',str(code/'autoresearch/configs'/state.get('protocol_name','robofollow_eval.json')),'--planner-mode',mode]
    remaining=state['candidate']['timeout_seconds']-elapsed_record(state['training'])
    if remaining<=0:raise RuntimeError('candidate total wall-time limit exhausted during evaluation preparation')
    state['status']='evaluating';atomic_json(run/'run.json',state)
    try:
        result=run_process(argv,run/phase_name,registry,remaining,2,env=launch_environment(),cwd=code)
        state['evaluation']=result
        state['status']='completed' if result['status']=='completed' else result['status']
    except BaseException:
        state['status']='interrupted';raise
    finally:atomic_json(run/'run.json',state)
    return state


def review_run(run,registry):
    from autoresearch.run_eval import collect_pair
    from autoresearch.evaluation import calibrate_thresholds,compare_candidate
    state=json.loads((run/'run.json').read_text())
    if state['status']!='completed':raise ValueError('review requires complete paired development evaluations')
    if state['candidate'].get('data_version','frozen-v1')!='complete-v3':raise ValueError('superseded split: metrics are diagnostic only')
    protocol=json.loads((run/state.get('active_evaluation_code','evaluation_code')/'autoresearch/configs'/state.get('protocol_name','robofollow_eval.json')).read_text())
    metrics,rows=collect_pair(run/state.get('active_evaluation_directory','paired_eval'),protocol)
    # Validate the complete payload before freezing any baseline or best artifact.
    json.dumps(metrics,allow_nan=False)
    baseline_path=registry.root/'baseline.json';thresholds_path=registry.root/'thresholds.json'
    if state['candidate']['id']=='R0':
        if baseline_path.exists() and json.loads(baseline_path.read_text())['run_id']!=run.name:raise ValueError('baseline is already frozen')
        thresholds=calibrate_thresholds(rows['correct'])
        atomic_json(thresholds_path,dict(thresholds,protocol_hash=protocol['hash'],baseline_run=run.name))
        atomic_json(baseline_path,dict(run_id=run.name,metrics=metrics))
        decision=dict(decision='baseline',reason='calibration only; no improvement claim')
        if not (registry.root/'best.json').exists():atomic_json(registry.root/'best.json',dict(run_id=run.name,metrics=metrics))
    else:
        baseline=json.loads(baseline_path.read_text());thresholds=json.loads(thresholds_path.read_text())
        if thresholds['protocol_hash']!=protocol['hash']:raise ValueError('frozen threshold protocol mismatch')
        decision=compare_candidate(baseline['metrics']['correct'],metrics['correct'],thresholds)
        if decision['decision']=='keep':
            best=json.loads((registry.root/'best.json').read_text())
            if metrics['correct']['mean_intent_score']>best['metrics']['correct']['mean_intent_score']:atomic_json(registry.root/'best.json',dict(run_id=run.name,metrics=metrics))
    report=dict(run_id=run.name,candidate=state['candidate'],hypothesis=state.get('hypothesis','baseline run prepared before hypothesis registry'),metrics=metrics,admission=decision,next_suggestion=next_suggestion(metrics,decision))
    atomic_json(run/'evidence.json',report)
    return report

def next_suggestion(metrics,decision):
    correct=metrics['correct']
    if correct['mean_intent_score']==0 and correct['mean_exec_score']==0:
        return 'inspect action/state scaling, 14D projection initialization and small-batch fitting before attributing failure to language'
    if decision['decision']=='reject':return 'inspect action feasibility and normalization; revert the regressing candidate'
    if metrics['language_delta']['mean_intent_score']<=0:
        return 'instruction reliance is not demonstrated; inspect conditioning gradients and matched instruction switching'
    if decision['decision']=='keep':return 'confirm on held-out levels/tasks and more seeds before claiming instruction-following generalization'
    return 'inspect predicted/teacher feature gap and scene failures; propose one factor with the frozen protocol'

def train_run(run,registry):
    state=json.loads((run/'run.json').read_text())
    if state['status']!='prepared':raise ValueError('run is not prepared; retries require a new run')
    c=state['candidate'];validate_candidate(c);code=run/'code'
    if c.get('data_version','frozen-v1')!='complete-v3':raise ValueError('superseded split: new training requires complete-v3')
    import yaml
    config=yaml.safe_load((code/'ge_act/configs/ltx_model/robofollow'/candidate_config(c['variant'],c.get('data_version','frozen-v1'))).read_text())
    require_image_validation(config['data']['train']['manifest_path'])
    config['output_dir']=str(run/'checkpoints')
    config_path=code/'ge_act/configs/ltx_model/robofollow/run.yaml'
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    state['effective_config_sha256']=file_sha256(config_path)
    argv=[PIX,'run','--manifest-path',str(code/'autoresearch/runtime/pixi.toml'),'train','--config_file','configs/ltx_model/robofollow/run.yaml','--max_train_steps',str(c['steps'])]
    state['status']='running';atomic_json(run/'run.json',state)
    env=launch_environment(CUDA_VISIBLE_DEVICES='0')
    try:
        result=run_process(argv,run/'train',registry,c['timeout_seconds'],1,env=env,cwd=code)
        state['status']='trained' if result['status']=='completed' else result['status']
        state['training']=result
    except BaseException:
        state['status']='interrupted';raise
    finally:atomic_json(run/'run.json',state)
    return state


def run_cycle(candidate,root,registry):
    run=prepare_run(candidate,root)
    state=train_run(run,registry)
    if state['status']!='trained':return dict(run_id=run.name,status=state['status'])
    state=evaluate_run(run,registry)
    if state['status']!='completed':return dict(run_id=run.name,status=state['status'])
    return review_run(run,registry)

def candidate_reservation(candidate):
    validate_candidate(candidate)
    return 2*candidate['timeout_seconds']/3600

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--data-version',choices=['frozen-v1','complete-v2','complete-v3'],default='complete-v3')
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--candidate',choices=['R0','R1','R2','R3'],default='R0');prep.add_argument('--steps',type=int,default=300);prep.add_argument('--timeout-seconds',type=int,default=7200)
    for name in ('run','recover','review','evaluate'):
        s=sub.add_parser(name);s.add_argument('--run-id',required=True)
    loop=sub.add_parser('loop');loop.add_argument('--candidates',nargs='+',choices=['R0','R1','R2','R3'],default=['R0','R1','R2','R3']);loop.add_argument('--steps',type=int,default=300);loop.add_argument('--timeout-seconds',type=int,default=7200)
    sub.add_parser('status');a=p.parse_args();registry=Registry(a.root)
    if a.command=='status':
        print(json.dumps(dict(used_gpu_hours=registry.used_gpu_hours(),max_gpu_hours=registry.max_gpu_hours,runs=[json.loads(x.read_text()) for x in a.root.glob('*/run.json')]),indent=2));return
    with registry.lock(),termination_cleanup():
        if a.command=='loop':
            reports=[]
            for name in a.candidates:
                if name=='R0' and (a.root/'baseline.json').exists():continue
                variant={'R0':'text','R1':'predicted','R2':'mix','R3':'joint'}[name]
                candidate=dict(id=name,variant=variant,steps=a.steps,timeout_seconds=a.timeout_seconds,data_version=a.data_version)
                try:registry.check_budget(candidate_reservation(candidate))
                except RuntimeError:
                    reports.append(dict(status='budget_stop',next_candidate=name,used_gpu_hours=registry.used_gpu_hours(),reason='cannot reserve candidate timeout times two GPUs'));atomic_json(a.root/'loop_summary.json',reports);break
                report=run_cycle(candidate,a.root,registry)
                reports.append(report);atomic_json(a.root/'loop_summary.json',reports)
                if report.get('status') in ('failed','interrupted'):break
                if name!='R0' and report['metrics']['correct']['mean_intent_score']==0 and report['metrics']['correct']['mean_exec_score']==0:
                    reports.append(dict(status='evidence_stop',reason=report['next_suggestion'],next_candidate='action baseline diagnostic'));atomic_json(a.root/'loop_summary.json',reports);break
            print(json.dumps(reports,indent=2));return
        if a.command=='prepare':
            variant={'R0':'text','R1':'predicted','R2':'mix','R3':'joint'}[a.candidate]
            print(prepare_run(dict(id=a.candidate,variant=variant,steps=a.steps,timeout_seconds=a.timeout_seconds,data_version=a.data_version),a.root));return
        if a.command=='status':
            print(json.dumps(dict(used_gpu_hours=registry.used_gpu_hours(),max_gpu_hours=registry.max_gpu_hours,runs=[json.loads(x.read_text()) for x in a.root.glob('*/run.json')]),indent=2));return
        run=(a.root/a.run_id).resolve()
        if run.parent!=a.root.resolve():raise ValueError('run-id must be a direct run directory')
        if a.command=='recover':print(json.dumps(recover_run(run,registry)));return
        state=json.loads((run/'run.json').read_text())
        if a.command=='review':
            print(json.dumps(review_run(run,registry),indent=2));return
        if a.command=='evaluate':
            print(json.dumps(evaluate_run(run,registry),indent=2));return
        print(json.dumps(train_run(run,registry),indent=2))

if __name__=='__main__':main()
