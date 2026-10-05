"""RoboFollow absolute-joint contract; raw pairs shift exactly once."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import importlib.util
import json
import os
import h5py
import numpy as np

CAMERA_ORDER = ['head', 'left_wrist', 'right_wrist']
RAW_JOINTS = ('left_arm','left_gripper','right_arm','right_gripper')
PAIRED_JOINTS = ('left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states')
RAW_CAMERAS = ('head_camera','left_camera','right_camera')
PAIRED_CAMERAS = ('cam_head','cam_left_wrist','cam_right_wrist')
JOINT_ORDER = ['left_arm_0','left_arm_1','left_arm_2','left_arm_3','left_arm_4','left_arm_5','left_gripper','right_arm_0','right_arm_1','right_arm_2','right_arm_3','right_arm_4','right_arm_5','right_gripper']

@dataclass
class Episode:
    images: np.ndarray | None
    states: np.ndarray
    actions: np.ndarray
    instructions: list[str]
    schema: str
    identity: tuple[str,str,str]

@lru_cache(maxsize=1)
def codec():
    root = Path(os.environ.get('ROBOFOLLOW_ROOT','/data/users/junjie/workspace/robofollow/RoboTwin'))
    spec = importlib.util.spec_from_file_location('robofollow_rgb_codec',root/'data/decode_image_bit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def joint_vector(group, names):
    result=[]
    for name,width in zip(names,(6,1,6,1)):
        a=np.asarray(group[name],dtype=np.float32)
        if a.ndim==1 and width==1:a=a[:,None]
        if a.ndim!=2 or a.shape[1]!=width or not np.isfinite(a).all():
            raise ValueError(f'invalid joint {name}: {a.shape}')
        result.append(a)
    if len({len(a) for a in result})!=1:raise ValueError('joint lengths differ')
    return np.concatenate(result,axis=1)

def read_episode(path: Path, image_indices=None, decode_images=True) -> Episode:
    path=Path(path)
    with h5py.File(path,'r') as f:
        raw='joint_action' in f and 'observation' in f
        paired=all(k in f for k in ('state','action','vision'))
        if raw==paired:raise ValueError('expected exactly one raw or paired schema')
        if raw:
            x=joint_vector(f['joint_action'],RAW_JOINTS)
            states,actions=x[:-1],x[1:]
            cams=[f[f'observation/{c}/rgb'] for c in RAW_CAMERAS]
            schema='robofollow-raw-v1'; camera_length=len(x)
        else:
            states=joint_vector(f['state'],PAIRED_JOINTS)
            actions=joint_vector(f['action'],PAIRED_JOINTS)
            cams=[f[f'vision/{c}/colors'] for c in PAIRED_CAMERAS]
            schema='robotwin-2026-09'; camera_length=len(states)
        if not len(states) or states.shape!=actions.shape:raise ValueError('empty/unaligned joint pairs')
        if any(c.ndim!=1 or len(c)!=camera_length for c in cams):raise ValueError('camera length/shape mismatch')
        if 'instructions' in f:instructions=json.loads(f['instructions'].asstr()[()])
        else:instructions=json.loads((path.parent.parent/'instructions'/f'{path.stem}.json').read_text())['seen']
        if not isinstance(instructions,list) or not instructions or not all(isinstance(t,str) and t.strip() for t in instructions):raise ValueError('invalid instructions')
        images=None
        if decode_images:
            idx=np.arange(len(states)) if image_indices is None else np.asarray(image_indices,dtype=np.int64)
            if idx.ndim!=1 or not len(idx) or (idx<0).any() or (idx>=len(states)).any():raise ValueError('invalid frame indices')
            images=np.stack([np.stack([codec().decode_image_bit(c[int(i)]) for c in cams]) for i in idx])
            if images.dtype!=np.uint8 or images.ndim!=5 or images.shape[-1]!=3:raise ValueError('invalid RGB frames')
    taskdir=path.parent.parent
    scene=taskdir.parent.name
    if scene not in ('scene1','scene2','scene3','scene4'):
        scene=taskdir.name.split('_')[0]
    task=taskdir.name.removeprefix(scene+'_')
    return Episode(images,states,actions,instructions,schema,(scene,task,path.stem))

def validate_stats(stats):
    if stats.get('action_type')!='absolute' or stats.get('action_space')!='joint' or stats.get('camera_order')!=CAMERA_ORDER or stats.get('joint_order')!=JOINT_ORDER:
        raise ValueError('statistics action/camera semantics mismatch')
    for kind in ('state','action'):
        mean=np.asarray(stats[f'{kind}_mean']);std=np.asarray(stats[f'{kind}_std'])
        if mean.shape!=(14,) or std.shape!=(14,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise ValueError('invalid statistics')

def normalize(x,stats,kind):
    validate_stats(stats)
    return (np.asarray(x,dtype=np.float32)-np.asarray(stats[f'{kind}_mean'],dtype=np.float32))/np.asarray(stats[f'{kind}_std'],dtype=np.float32)

def denormalize(x,stats,kind):
    validate_stats(stats)
    return np.asarray(x,dtype=np.float32)*np.asarray(stats[f'{kind}_std'],dtype=np.float32)+np.asarray(stats[f'{kind}_mean'],dtype=np.float32)


def validate_all_frames(path, batch_size=32, return_digest=False):
    """Stream every usable frame; retain only one decoded image in memory."""
    import hashlib
    digest=hashlib.sha256()
    e=read_episode(path,decode_images=False)
    with h5py.File(path,'r') as f:
        cams=[f[f'observation/{c}/rgb'] for c in RAW_CAMERAS] if e.schema=='robofollow-raw-v1' else [f[f'vision/{c}/colors'] for c in PAIRED_CAMERAS]
        expected=None
        for start in range(0,len(e.states),batch_size):
            for name,c in zip(CAMERA_ORDER,cams):
                for offset,payload in enumerate(c[start:min(start+batch_size,len(e.states))]):
                    i=start+offset
                    try:
                        image=codec().decode_image_bit(payload)
                        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3:raise ValueError('invalid RGB')
                        if expected is None:expected=image.shape
                        if image.shape!=expected:raise ValueError(f'image shape {image.shape} differs from {expected}')
                        if return_digest:
                            digest.update(f'{name}:{i}:{image.shape}'.encode());digest.update(image.tobytes())
                    except Exception as error:
                        raise ValueError(f'camera {name} frame {i}: {error}') from error
    count=len(e.states)*len(cams)
    return (count,digest.hexdigest()) if return_digest else count


def pad_offsets(native,size):
    """Top/left offsets that centre a native (h,w) frame inside a padded (H,W) canvas."""
    (h,w),(H,W)=native,size
    if h>H or w>W:raise ValueError(f'cannot pad {native} into {size}')
    return (H-h)//2,(W-w)//2

def pad_frames(x,size,value=-1.0):
    """Pad [...,h,w] frames (normalized to [-1,1]) symmetrically to size; -1 is black."""
    import torch.nn.functional as F
    top,left=pad_offsets(tuple(x.shape[-2:]),tuple(size))
    H,W=size
    return F.pad(x,(left,W-x.shape[-1]-left,top,H-x.shape[-2]-top),value=value)

def crop_padding(x,native,size):
    """Inverse of pad_frames on the last two axes."""
    top,left=pad_offsets(tuple(native),tuple(size))
    return x[...,top:top+native[0],left:left+native[1]]
