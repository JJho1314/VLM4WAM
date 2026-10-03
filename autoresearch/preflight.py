"""Build immutable training-domain manifests and train-only statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ge_act.data.robofollow_schema import CAMERA_ORDER,JOINT_ORDER,read_episode,validate_all_frames

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def assign_splits(episodes, seed):
    """Keep task/instruction groups and all trajectory aliases in one component."""
    parents={};trajectory_group={}
    def root(group):
        parents.setdefault(group,group)
        while parents[group]!=group:
            parents[group]=parents[parents[group]];group=parents[group]
        return group
    def group_of(row):
        return f"{row['scene']}:{row['task']}:"+json.dumps(sorted(row['instructions']))
    for row in episodes:
        group=group_of(row);root(group)
        previous=trajectory_group.setdefault(row['trajectory_hash'],group)
        x,y=root(group),root(previous)
        if x!=y:parents[max(x,y)]=min(x,y)
    for row in episodes:
        key=root(group_of(row))
        row['split_group']=key
        row['split']='dev' if int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:8],16)%5==0 else 'train'

def build_manifest(root: Path, output: Path, seed: int=42, training_registry=None) -> dict:
    result={'image_validation':'all usable frames, streaming RGB decode','version':2,'split_policy':'connected task/instruction and joint-trajectory groups','seed':seed,'root':str(Path(root).resolve()),'camera_order':CAMERA_ORDER,'joint_order':JOINT_ORDER,'episodes':[],'duplicates':[],'quarantine':[]}
    seen={}; identities={}
    for p in sorted(Path(root).glob('**/data/episode*.hdf5')):
        try:
            e=read_episode(p,decode_images=False)
            _,image_hash=validate_all_frames(p,return_digest=True)
            scene,task,ep=e.identity
            if scene not in ('scene1','scene2','scene3','scene4'):raise ValueError('unknown scene')
            if training_registry is not None:
                expected=training_registry[scene][task]
                if e.instructions!=expected:raise ValueError('instructions differ from official training registry')
            # Preserve instruction aliases, but connect their complete task groups before splitting.
            trajectory=hashlib.sha256(e.states.tobytes()+e.actions.tobytes()).hexdigest()
            content=hashlib.sha256(e.states.tobytes()+e.actions.tobytes()+json.dumps(e.instructions).encode()+image_hash.encode()).hexdigest()
            identity=':'.join(e.identity)
            if content in seen or identity in identities:
                result['duplicates'].append({'path':str(p),'same_as':seen.get(content,identities.get(identity))});continue
            seen[content]=str(p);identities[identity]=str(p)
            group=f'{scene}:{task}:'+json.dumps(sorted(e.instructions))
            split='dev' if int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:8],16)%5==0 else 'train'
            result['episodes'].append({'path':str(p.resolve()),'identity':identity,'scene':scene,'task':task,'instructions':e.instructions,'schema':e.schema,'length':len(e.states),'content_hash':content,'trajectory_hash':trajectory,'file_hash':digest(p),'image_hash':image_hash,'split':split,'sample_indices':[0,len(e.states)//2,len(e.states)-1]})
        except (OSError,ValueError,KeyError,TypeError) as error:
            result['quarantine'].append({'path':str(p),'reason':str(error)})
        if (len(result['episodes'])+len(result['quarantine'])+len(result['duplicates']))%100==0:
            print(f"scanned {len(result['episodes'])} usable episodes",flush=True)
    if not result['episodes']:raise ValueError('no usable episodes')
    assign_splits(result['episodes'],seed)
    write_json(output,result)
    return result

def compute_statistics(manifest: Path, output: Path) -> dict:
    m=json.loads(Path(manifest).read_text());sums={k:np.zeros(14,dtype=np.float64) for k in ('state','action')};squares={k:np.zeros(14,dtype=np.float64) for k in sums};mins={k:np.full(14,np.inf) for k in sums};maxs={k:np.full(14,-np.inf) for k in sums};count=0
    for row in m['episodes']:
        if row['split']!='train':continue
        e=read_episode(Path(row['path']),decode_images=False);count+=len(e.states)
        for kind,x in [('state',e.states),('action',e.actions)]:
            x=x.astype(np.float64);sums[kind]+=x.sum(0);squares[kind]+=(x*x).sum(0);mins[kind]=np.minimum(mins[kind],x.min(0));maxs[kind]=np.maximum(maxs[kind],x.max(0))
    if not count:raise ValueError('empty training split')
    s={'version':1,'manifest_hash':digest(manifest),'action_type':'absolute','action_space':'joint','camera_order':CAMERA_ORDER,'joint_order':JOINT_ORDER,'num_pairs':count}
    for k in sums:
        mean=sums[k]/count;std=np.sqrt(np.maximum(squares[k]/count-mean*mean,0));active=std>1e-6
        s.update({f'{k}_mean':mean.tolist(),f'{k}_std':np.where(active,std,1).tolist(),f'{k}_active':active.tolist(),f'{k}_min':mins[k].tolist(),f'{k}_max':maxs[k].tolist()})
    write_json(output,s);return s

def official_training_registry(root):
    import sys
    sys.path.insert(0,str(Path(root).resolve()))
    from robofollow.registry import training
    return {scene:training(scene)[1] for scene in ('scene1','scene2','scene3','scene4')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--robofollow-root',default='/data/users/junjie/workspace/robofollow/RoboTwin');p.add_argument('--stats-only',action='store_true');a=p.parse_args()
    out=Path(a.output)
    if not a.stats_only:build_manifest(Path(a.root),out/'manifest.json',training_registry=official_training_registry(a.robofollow_root))
    compute_statistics(out/'manifest.json',out/'stats.json')
    print('manifest and train-only statistics ready',flush=True)
if __name__=='__main__':main()
