import json
import os
import subprocess
import pytest

def test_budget_and_lock(tmp_path):
    from autoresearch.registry import Registry
    r=Registry(tmp_path,max_gpu_hours=8)
    with r.lock():
        with pytest.raises(RuntimeError):
            with Registry(tmp_path).lock():pass
    r.charge('a',3600,2)
    assert r.used_gpu_hours()==2
    r.charge('b',10800,2)
    with pytest.raises(RuntimeError):r.check_budget(1)

def test_argv_whitelist():
    from autoresearch.controller import validate_candidate
    with pytest.raises(ValueError):validate_candidate({'id':'x','shell':'rm -rf /'})
    with pytest.raises(ValueError):validate_candidate({'id':'R0','variant':'text','steps':100,'timeout_seconds':8000})
    validate_candidate({'id':'R0','variant':'text','steps':100,'timeout_seconds':7200})

def test_process_identity():
    from autoresearch.registry import process_identity,safe_signal
    identity=process_identity(os.getpid())
    assert identity['pid']==os.getpid()
    forged=dict(identity,start_ticks=identity['start_ticks']+1)
    assert safe_signal(forged,0) is False

def test_timeout_and_failure(tmp_path):
    from autoresearch.controller import run_process
    from autoresearch.registry import Registry
    r=Registry(tmp_path)
    record=run_process(['sleep','3'],tmp_path/'slow',r,timeout_seconds=.1,gpus=1)
    assert record['status']=='interrupted' and record['gpu_hours']>0
    record=run_process(['false'],tmp_path/'bad',r,timeout_seconds=1,gpus=0)
    assert record['status']=='failed'

def test_snapshot_preserves_dirty_and_untracked(tmp_path):
    from autoresearch.controller import snapshot_code
    src=tmp_path/'repo';src.mkdir()
    subprocess.run(['git','init',str(src)],check=True,capture_output=True)
    (src/'a').write_text('base')
    subprocess.run(['git','-C',str(src),'add','a'],check=True)
    subprocess.run(['git','-C',str(src),'-c','user.name=Test','-c','user.email=test@local','commit','-m','base'],check=True,capture_output=True)
    (src/'a').write_text('dirty');(src/'provider.py').write_text('untracked')
    dst=tmp_path/'snap';snapshot_code(src,dst)
    assert (dst/'a').read_text()=='dirty' and (dst/'provider.py').read_text()=='untracked'
    assert (dst/'.snapshot.json').exists()

def test_snapshot_captures_gitlink_directory(tmp_path):
    from autoresearch.controller import snapshot_code
    src=tmp_path/'repo';src.mkdir()
    subprocess.run(['git','init',str(src)],check=True,capture_output=True)
    sub=src/'vendor';sub.mkdir()
    subprocess.run(['git','init',str(sub)],check=True,capture_output=True)
    (sub/'x.py').write_text('nested')
    subprocess.run(['git','-C',str(sub),'add','.'],check=True)
    subprocess.run(['git','-C',str(sub),'-c','user.name=Test','-c','user.email=test@local','commit','-m','nested'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(src),'add','.'],check=True,capture_output=True)
    subprocess.run(['git','-C',str(src),'-c','user.name=Test','-c','user.email=test@local','commit','-m','base'],check=True,capture_output=True)
    dst=tmp_path/'snap';snapshot_code(src,dst)
    assert (dst/'vendor/x.py').read_text()=='nested'

def test_partial_review_never_updates_best(tmp_path):
    from autoresearch.controller import review_run
    from autoresearch.registry import Registry
    r=Registry(tmp_path);run=tmp_path/'R0';run.mkdir()
    (run/'run.json').write_text(json.dumps(dict(status='trained',candidate={'id':'R0'})))
    with pytest.raises(ValueError):review_run(run,r)
    assert not (tmp_path/'best.json').exists()

def test_recovery_charges_once_and_does_not_signal_reused_pid(tmp_path):
    from autoresearch.controller import recover_run
    from autoresearch.registry import Registry,process_identity
    r=Registry(tmp_path);run=tmp_path/'R0';run.mkdir();phase=run/'train';phase.mkdir()
    (run/'run.json').write_text(json.dumps(dict(status='running')))
    identity=process_identity(os.getpid());identity['start_ticks']+=1
    import time
    (phase/'process.json').write_text(json.dumps(dict(status='running',started_at=time.time()-10,gpus=1,process=identity)))
    recover_run(run,r);first=r.used_gpu_hours();recover_run(run,r)
    assert r.used_gpu_hours()==first and first>0

def test_readiness_failure_and_protocol_digest(tmp_path):
    from autoresearch.run_eval import wait_service,verify_protocol
    proc=subprocess.Popen(['false']);proc.wait()
    with pytest.raises(RuntimeError):wait_service(proc,'127.0.0.1',8999,timeout=.1)
    with pytest.raises(ValueError):verify_protocol(dict(hash='forged',instruction_modes=['correct','shuffle']))

def test_owned_process_tree_cleanup():
    from autoresearch.registry import descendant_identities,stop_process_tree,process_identity
    import time
    proc=subprocess.Popen(['sh','-c','sleep 30 & wait'])
    time.sleep(.1)
    ids=descendant_identities(proc.pid)
    assert len(ids)>=2
    stop_process_tree(proc,timeout=1)
    assert proc.poll() is not None
    from pathlib import Path
    for identity in ids:
        if process_identity(identity['pid'])==identity:
            assert Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')',1)[1].split()[0]=='Z'

def test_registered_variant_configs_and_frozen_admission(tmp_path):
    from autoresearch.controller import candidate_config, require_admission
    assert candidate_config('text')=='action_model_text.yaml'
    assert candidate_config('mix')=='action_model_mix.yaml'
    with pytest.raises(ValueError):candidate_config('shell')
    marker=tmp_path/'validation.json'
    marker.write_text(json.dumps({'validated_variants':['predicted','mix','joint']}))
    with pytest.raises(ValueError,match='baseline'):
        require_admission('predicted',tmp_path,marker)
    (tmp_path/'baseline.json').write_text('{}')
    (tmp_path/'thresholds.json').write_text('{}')
    require_admission('predicted',tmp_path,marker)

def test_review_nan_cannot_create_best(tmp_path,monkeypatch):
    from autoresearch.controller import review_run
    from autoresearch.registry import Registry
    import autoresearch.run_eval as evaluation
    import autoresearch.evaluation as scoring
    run=tmp_path/'run';run.mkdir();code=run/'evaluation_code/autoresearch/configs';code.mkdir(parents=True)
    (code/'robofollow_eval.json').write_text(json.dumps({'hash':'p'}))
    (run/'run.json').write_text(json.dumps(dict(status='completed',candidate={'id':'R0','data_version':'complete-v3'})))
    monkeypatch.setattr(evaluation,'collect_pair',lambda *a:({'correct':{'mean_intent_score':float('nan')}},{'correct':[]}))
    monkeypatch.setattr(scoring,'calibrate_thresholds',lambda *a:dict(version=1,min_intent_gain=.01))
    with pytest.raises(ValueError):review_run(run,Registry(tmp_path))
    assert not (tmp_path/'best.json').exists()
    assert not (tmp_path/'thresholds.json').exists()

@pytest.mark.parametrize('record_exists',[True,False])
def test_evaluation_retry_preserves_prior_attempt(tmp_path,monkeypatch,record_exists):
    import time
    import autoresearch.controller as c
    from autoresearch.registry import Registry
    run=tmp_path/'run';run.mkdir()
    checkpoint=run/'checkpoints/stamp/step_1';checkpoint.mkdir(parents=True)
    (checkpoint/'diffusion_pytorch_model.safetensors').write_bytes(b'probe')
    prior=run/'evaluation';prior.mkdir();(prior/'process.log').write_text('failure evidence')
    old=dict(status='failed',process={'pid':123})
    state=dict(status='failed',candidate=dict(variant='text',steps=1,timeout_seconds=7200),training=dict(status='completed',started_at=time.time()))
    if record_exists:state['evaluation']=old
    (run/'run.json').write_text(json.dumps(state))
    monkeypatch.setattr(c,'bind_run_checkpoint',lambda *a:None)
    monkeypatch.setattr(c,'snapshot_code',lambda src,dst:dst.mkdir())
    seen=[]
    def execute(argv,directory,*a,**kw):
        seen.append((argv,directory));return dict(status='completed',process={'pid':456})
    monkeypatch.setattr(c,'run_process',execute)
    state=c.evaluate_run(run,Registry(tmp_path))
    if record_exists:assert state['evaluation_attempts']==[old]
    assert state['active_evaluation_directory']=='paired_eval_attempt_2'
    assert seen[0][1].name=='evaluation_attempt_2'
    assert (prior/'process.log').read_text()=='failure evidence'

def test_cycle_stops_after_training_failure(tmp_path,monkeypatch):
    import autoresearch.controller as c
    from autoresearch.registry import Registry
    monkeypatch.setattr(c,'prepare_run',lambda *a:tmp_path/'R1')
    monkeypatch.setattr(c,'train_run',lambda *a:dict(status='failed'))
    monkeypatch.setattr(c,'evaluate_run',lambda *a:pytest.fail('must not evaluate failed training'))
    assert c.run_cycle({},tmp_path,Registry(tmp_path))==dict(run_id='R1',status='failed')

def test_data_version_changes_config_and_blocks_mixed_baseline(tmp_path):
    from autoresearch.controller import candidate_config,require_admission,validate_candidate
    assert candidate_config('text','complete-v2')=='action_model_text_complete_v2.yaml'
    validate_candidate(dict(id='R0',variant='text',steps=1,timeout_seconds=60,data_version='complete-v2'))
    with pytest.raises(ValueError):candidate_config('text','unknown')
    (tmp_path/'baseline.json').write_text(json.dumps(dict(metrics=dict(correct=dict(protocol_hash='v1')))))
    (tmp_path/'thresholds.json').write_text(json.dumps(dict(protocol_hash='v1')))
    marker=tmp_path/'marker.json';marker.write_text(json.dumps(dict(validated_variants=['predicted'])))
    with pytest.raises(ValueError,match='protocol'):
        require_admission('predicted',tmp_path,marker,protocol_hash='v2')

def test_timeout_and_billing_survive_wall_clock_freeze(tmp_path,monkeypatch):
    import autoresearch.controller as c
    from autoresearch.registry import Registry
    monkeypatch.setattr(c.time,'time',lambda:0.)
    record=c.run_process(['sleep','.3'],tmp_path/'phase',Registry(tmp_path),.05,1)
    assert record['status']=='interrupted' and record['gpu_hours']>0

def test_next_suggestion_uses_action_and_language_evidence():
    from autoresearch.controller import next_suggestion
    metrics=dict(correct=dict(mean_intent_score=0.,mean_exec_score=0.),language_delta=dict(mean_intent_score=0.))
    assert 'action' in next_suggestion(metrics,{'decision':'baseline'})
    metrics['correct'].update(mean_intent_score=.5,mean_exec_score=.5)
    assert 'instruction' in next_suggestion(metrics,{'decision':'keep'})
    metrics['language_delta']['mean_intent_score']=.4
    assert 'held-out' in next_suggestion(metrics,{'decision':'keep'})

def test_failed_launcher_cleans_owned_orphan(tmp_path):
    from autoresearch.controller import run_process
    from autoresearch.registry import Registry,process_identity
    from pathlib import Path
    phase=tmp_path/'phase'
    record=run_process(['sh','-c','sleep 30 & echo $!; sleep .1; exit 1'],phase,Registry(tmp_path),2,0)
    child=int((phase/'process.log').read_text().strip())
    identity=process_identity(child)
    try:
        assert record['status']=='failed'
        if identity is not None:assert Path(f'/proc/{child}/stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
    finally:
        from autoresearch.registry import safe_signal
        import signal
        safe_signal(identity,signal.SIGKILL)

def test_corrected_data_and_smaller_timeout_reservation(tmp_path):
    from autoresearch.controller import candidate_config,candidate_reservation,prepare_run
    assert candidate_config('text','complete-v3')=='action_model_text_complete_v3.yaml'
    assert candidate_reservation(dict(id='R0',variant='text',steps=300,timeout_seconds=4500,data_version='complete-v3'))==2.5
    with pytest.raises(ValueError,match='superseded'):
        prepare_run(dict(id='R0',variant='text',steps=300,timeout_seconds=4500,data_version='frozen-v1'),tmp_path)

def test_registered_full_data_paths_keep_storage_root():
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[1]
    for version in ['complete_v2','complete_v3']:
        for variant in ['text','planner','mix','joint']:
            cfg=yaml.safe_load((root/f'ge_act/configs/ltx_model/robofollow/action_model_{variant}_{version}.yaml').read_text())
            for split in ['train','val']:
                for key in ['manifest_path','stat_file']:
                    path=Path(cfg['data'][split][key])
                    assert path.parts[:3]==('/','data','users')
                    assert path.parent.name=='data_'+version

def test_paired_workers_overlap_and_stop_on_failure(tmp_path):
    from autoresearch.run_eval import run_mode_workers
    import sys,time
    # A worker can finish only after the other is running: serial launch fails.
    script="import pathlib,time,sys; a,b=map(pathlib.Path,sys.argv[1:]); a.write_text('ready'); end=time.monotonic()+2;\nwhile not b.exists() and time.monotonic()<end: time.sleep(.01)\nassert b.exists()"
    a,b=tmp_path/'a',tmp_path/'b'
    run_mode_workers([[sys.executable,'-c',script,str(a),str(b)],[sys.executable,'-c',script,str(b),str(a)]],tmp_path/'logs')
    assert a.exists() and b.exists()
    start=time.monotonic()
    with pytest.raises(subprocess.CalledProcessError):run_mode_workers([['false'],['sleep','30']],tmp_path/'failed')
    assert time.monotonic()-start<3

def test_research_environment_sanitized(monkeypatch):
    from autoresearch.controller import launch_environment
    monkeypatch.setenv('BATON_RESEARCH_CAPTION_DROPOUT','1')
    monkeypatch.setenv('BATON_RESEARCH_JOINT_HEAD_LR','99')
    env=launch_environment()
    assert env['BATON_RESEARCH_CAPTION_DROPOUT']=='0'
    assert 'BATON_RESEARCH_JOINT_HEAD_LR' not in env

def test_unresolved_gpu_phase_prevents_new_launch(tmp_path):
    from autoresearch.controller import refuse_unresolved
    from autoresearch.registry import Registry,atomic_json
    atomic_json(tmp_path/'old/train/process.json',{'status':'running'})
    with pytest.raises(RuntimeError,match='unresolved'):refuse_unresolved(Registry(tmp_path))

@pytest.mark.parametrize('kill_signal',[15,9])
def test_controller_death_cleanup_and_launch_exclusion(tmp_path,kill_signal):
    import signal,time
    from pathlib import Path
    from autoresearch.controller import PIX,ROOT,refuse_unresolved
    from autoresearch.registry import Registry,process_identity
    script="from pathlib import Path; from autoresearch.controller import run_process,termination_cleanup; from autoresearch.registry import Registry; import sys; root=Path(sys.argv[1]); guard=termination_cleanup(); guard.__enter__(); run_process(['sleep','60'],root/'run/train',Registry(root),5,1)"
    source=tmp_path/'controller_child.py';source.write_text(script)
    log=(tmp_path/'controller.log').open('w')
    launcher=subprocess.Popen([PIX,'run','--manifest-path',str(ROOT/'autoresearch/runtime/pixi.toml'),'research',str(source),str(tmp_path)],stdout=log,stderr=subprocess.STDOUT)
    path=tmp_path/'run/train/process.json'
    deadline=time.monotonic()+15
    try:
        while not path.exists() and time.monotonic()<deadline:time.sleep(.05)
        assert path.exists(),(tmp_path/'controller.log').read_text()
        record=json.loads(path.read_text());target=record['process']
        os.kill(record['controller']['pid'],kill_signal)
        launcher.wait(timeout=10)
        # SIGKILL may leave a running record until independent owner-loss enforcement.
        if json.loads(path.read_text())['status']=='running':
            with pytest.raises(RuntimeError,match='unresolved'):refuse_unresolved(Registry(tmp_path))
        while json.loads(path.read_text())['status']=='running' and time.monotonic()<deadline:time.sleep(.1)
        assert json.loads(path.read_text())['status']=='interrupted'
        assert Registry(tmp_path).used_gpu_hours()>0
        current=process_identity(target['pid'])
        if current==target:
            assert Path(f"/proc/{target['pid']}/stat").read_text().rsplit(')',1)[1].split()[0]=='Z'
        refuse_unresolved(Registry(tmp_path))
    finally:
        if launcher.poll() is None:launcher.kill();launcher.wait()
