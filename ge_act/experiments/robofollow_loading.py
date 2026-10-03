"""Explicit checkpoint transfer, with no blanket strict=False omissions."""
from pathlib import Path
import hashlib
import json
from safetensors.torch import load_file

def load_robofollow_weights(model,checkpoint: Path,allowed_new_keys: set[str]) -> dict:
    state=load_file(str(checkpoint),device='cpu');target=model.state_dict()
    unexpected=set(state)-set(target)
    if unexpected:raise ValueError(f'unexpected checkpoint keys: {sorted(unexpected)}')
    missing=set(target)-set(state)
    mismatch={k for k in state if state[k].shape!=target[k].shape}
    if mismatch-allowed_new_keys:raise ValueError(f'shape mismatch: {sorted(mismatch-allowed_new_keys)}')
    if missing-allowed_new_keys:raise ValueError(f'missing keys: {sorted(missing-allowed_new_keys)}')
    # Allowed topology changes initialize in the destination; matching source layers still transfer.
    loaded={k:v for k,v in state.items() if k not in mismatch}
    model.load_state_dict(loaded,strict=False)
    topology=[(k,list(v.shape)) for k,v in target.items()]
    return {'checkpoint':str(checkpoint),'loaded':sorted(loaded),'new':sorted(missing|mismatch),'topology_hash':hashlib.sha256(json.dumps(topology).encode()).hexdigest()}

def ensure_vae_channels(vae):
    channels=int(getattr(vae.config,'latent_channels',0))
    if channels<=0:raise ValueError('VAE config lacks valid latent channels')
    if hasattr(vae,'z_dim') and vae.z_dim!=channels:raise ValueError('VAE latent channel aliases disagree')
    vae.z_dim=channels


def weight_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def semantic_contract(config,stats_path):
    from ge_act.data.robofollow_schema import CAMERA_ORDER,JOINT_ORDER,validate_stats
    stats=json.loads(Path(stats_path).read_text());validate_stats(stats)
    md=config['robofollow_metadata']
    if md['camera_order']!=CAMERA_ORDER or md['action_type']!='absolute' or md['action_space']!='joint' or md['action_dim']!=14:
        raise ValueError('checkpoint semantic config mismatch')
    manifest=config['data']['train']['manifest_path']
    if weight_digest(manifest)!=stats['manifest_hash']:raise ValueError('checkpoint semantic manifest mismatch')
    pc=config.get('robofollow_planner',{})
    planner=None
    if pc.get('enabled'):
        root=Path(pc['checkpoint'])
        artifacts={str(x.relative_to(root)):weight_digest(x) for x in sorted(root.rglob('*')) if x.is_file()}
        if not artifacts:raise ValueError('empty planner checkpoint')
        planner=dict(config=pc,artifacts=artifacts)
    return dict(camera_order=CAMERA_ORDER,joint_order=JOINT_ORDER,action_type='absolute',action_space='joint',action_dim=14,statistics_sha256=weight_digest(stats_path),manifest_sha256=stats['manifest_hash'],history_action_stride=md['history_action_stride'],planner=planner)

def write_contract(checkpoint,semantics,audit):
    from autoresearch.registry import atomic_json
    path=Path(str(checkpoint)+'.contract.json')
    if path.exists():
        validate_contract(checkpoint,semantics);return
    atomic_json(path,dict(version=1,weights_sha256=weight_digest(checkpoint),semantics=semantics,audit=audit))

def validate_contract(checkpoint,expected):
    path=Path(str(checkpoint)+'.contract.json')
    if not path.exists():raise ValueError('missing checkpoint semantic contract')
    data=json.loads(path.read_text())
    if data.get('version')!=1 or data.get('semantics')!=expected:raise ValueError('checkpoint semantic contract mismatch')
    if data.get('weights_sha256')!=weight_digest(checkpoint):raise ValueError('checkpoint weights digest mismatch')
    return data
