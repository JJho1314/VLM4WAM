"""Lazy per-item RoboFollow reader compatible with GE Act generic batches."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from ge_act.data.robofollow_schema import CAMERA_ORDER,read_episode,normalize,validate_stats,pad_frames

class RoboFollowHDF5Dataset(Dataset):
    def __init__(self,manifest_path,stat_file,split=None,train_dataset=True,sample_size=(192,256),chunk=9,action_chunk=54,n_previous=4,action_type='absolute',action_space='joint',valid_cam=CAMERA_ORDER,source_fps=30,fix_sidx=None,resize_mode='resize',native_size=(240,320),hold_pad=0,**kwargs):
        if action_type!='absolute' or action_space!='joint' or list(valid_cam)!=CAMERA_ORDER:raise ValueError('RoboFollow requires ordered three cameras and absolute joints')
        if chunk!=9 or action_chunk!=54 or n_previous!=4:raise ValueError('initial RoboFollow temporal contract is 4 history + 54 actions / 9 future frames')
        self.manifest_path=Path(manifest_path);self.manifest=json.loads(self.manifest_path.read_text())
        self.stats=json.loads(Path(stat_file).read_text());validate_stats(self.stats)
        if self.stats.get('manifest_hash')!=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest():raise ValueError('statistics belong to a different manifest')
        self.records=[r for r in self.manifest['episodes'] if r['split']==(split or ('train' if train_dataset else 'dev'))]
        if not self.records:raise ValueError('empty selected split')
        if resize_mode not in ('resize','pad'):raise ValueError('resize_mode must be resize or pad')
        self.resize_mode=resize_mode;self.native_size=tuple(native_size)
        # Training windows may start up to hold_pad steps past the last frame; frame_indices clamps
        # them to the final pose, which teaches the policy to hold still after finishing.
        self.hold_pad=int(hold_pad)
        self.train_dataset=train_dataset;self.sample_size=tuple(sample_size);self.fix_sidx=fix_sidx;self.epoch=0;self.source_fps=source_fps
        self.n_previous=n_previous;self.chunk=chunk;self.action_chunk=action_chunk;self.action_dim=14
        self.action_mean={'robofollow':torch.tensor(self.stats['action_mean'])};self.action_std={'robofollow':torch.tensor(self.stats['action_std'])}
        self.state_mean={'robofollow':torch.tensor(self.stats['state_mean'])};self.state_std={'robofollow':torch.tensor(self.stats['state_std'])}
    def __len__(self):return len(self.records)
    def set_epoch(self,epoch):self.epoch=int(epoch)
    @staticmethod
    def frame_indices(t,length):
        history=np.clip(np.array([-150,-100,-50,0])+t,0,length-1).tolist()
        actions=np.clip(np.arange(54)+t,0,length-1).tolist()
        return history+actions[5::6],history+actions
    def __getitem__(self,index):
        r=self.records[index];length=r['length']
        seed=int(hashlib.sha256(f"{self.manifest.get('seed',42)}:{self.epoch}:{r['identity']}".encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed)
        t=self.fix_sidx if self.fix_sidx is not None else (int(rng.integers(length+self.hold_pad)) if self.train_dataset else length//2)
        frames,actions=self.frame_indices(t,length)
        e=read_episode(Path(r['path']),image_indices=frames)
        if e.schema!=r['schema'] or len(e.states)!=length or e.instructions!=r['instructions']:raise ValueError('episode no longer matches manifest')
        # No persistent file handle crosses worker boundaries; read_episode opens locally.
        images=torch.from_numpy(e.images).permute(0,1,4,2,3).float()/127.5-1
        shape=images.shape
        if self.resize_mode=='pad':
            if tuple(shape[-2:])!=self.native_size:raise ValueError(f'frame {tuple(shape[-2:])} != native {self.native_size}')
            images=pad_frames(images.reshape(-1,3,*shape[-2:]),self.sample_size)
        else:images=F.interpolate(images.reshape(-1,3,*shape[-2:]),size=self.sample_size,mode='bilinear',align_corners=False)
        video=images.reshape(shape[0],3,3,*self.sample_size).permute(2,1,0,3,4).contiguous()
        return {'video':video,'actions':torch.from_numpy(normalize(e.actions[actions],self.stats,'action')),'state':torch.from_numpy(normalize(e.states[[frames[3]]],self.stats,'state')),'caption':e.instructions[int(rng.integers(len(e.instructions))) if self.train_dataset else 0]}
