import json
from pathlib import Path
import cv2
import h5py
import numpy as np
import pytest


def episode(path, paired=False, offset=0, n=4, instruction='pick yellow'):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ('left_arm','left_gripper','right_arm','right_gripper')
    keys = ('left_arm_joint_states','left_ee_joint_states','right_arm_joint_states','right_ee_joint_states')
    with h5py.File(path, 'w') as f:
        f['instructions'] = json.dumps([instruction])
        for group in (('state','action') if paired else ('joint_action',)):
            for i,(key,w) in enumerate(zip(keys if paired else raw,(6,1,6,1))):
                values = np.arange(n)[:,None] + offset + i*10 + (100 if group=='action' else 0)
                f[f'{group}/{key}'] = np.repeat(values,w,axis=1).astype('f4')
        for v,cam in enumerate(('cam_head','cam_left_wrist','cam_right_wrist') if paired else ('head_camera','left_camera','right_camera')):
            img = np.zeros((8,10,3),np.uint8); img[:] = (40+v*30,20,10)
            ok, buf = cv2.imencode('.jpg',img)
            assert ok
            f[f'vision/{cam}/colors' if paired else f'observation/{cam}/rgb'] = np.array([buf.tobytes()]*n,dtype=f'S{len(buf)}')
    return path


def test_raw_aligns_once_and_camera_order(tmp_path):
    from ge_act.data.robofollow_schema import read_episode
    e = read_episode(episode(tmp_path/'scene1'/'task'/'data'/'episode0.hdf5'))
    assert e.states.shape == e.actions.shape == (3,14)
    np.testing.assert_array_equal(e.states[:,0],[0,1,2])
    np.testing.assert_array_equal(e.actions[:,0],[1,2,3])
    np.testing.assert_array_equal(e.states[0],[0]*6+[10]+[20]*6+[30])
    assert e.images.shape == (3,3,8,10,3)
    np.testing.assert_allclose(e.images[0,:,0,0,0],[40,70,100],atol=3)


def test_paired_never_shifts_again(tmp_path):
    from ge_act.data.robofollow_schema import read_episode
    e = read_episode(episode(tmp_path/'scene1'/'task'/'data'/'episode0.hdf5',paired=True))
    assert len(e.states)==4
    np.testing.assert_array_equal(e.actions[:,0],[100,101,102,103])


def test_manifest_quarantines_and_groups_duplicates(tmp_path):
    from autoresearch.preflight import build_manifest
    import shutil
    p=episode(tmp_path/'scene1'/'task'/'data'/'episode0.hdf5')
    shutil.copyfile(p,p.with_name('episode1.hdf5'))
    episode(tmp_path/'scene2'/'task2'/'data'/'episode0.hdf5',n=1)
    (p.parent/'episode2.hdf5').write_bytes(b'broken')
    m=build_manifest(tmp_path,tmp_path/'manifest.json')
    assert len(m['episodes'])==1 and len(m['duplicates'])==1 and len(m['quarantine'])==2
    assert m['episodes'][0]['instructions']==['pick yellow']


def test_statistics_train_only_roundtrip(tmp_path):
    from autoresearch.preflight import compute_statistics
    from ge_act.data.robofollow_schema import normalize, denormalize
    train=episode(tmp_path/'scene1'/'a'/'data'/'episode0.hdf5')
    dev=episode(tmp_path/'scene1'/'b'/'data'/'episode0.hdf5',offset=999)
    m={'episodes':[{'path':str(train),'split':'train'},{'path':str(dev),'split':'dev'}]}
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(m))
    s=compute_statistics(mp,tmp_path/'stats.json')
    assert s['state_mean'][0]==1 and s['action_mean'][0]==2
    x=np.arange(28,dtype='f4').reshape(2,14)
    np.testing.assert_allclose(denormalize(normalize(x,s,'action'),s,'action'),x,atol=1e-5)
    assert s['action_type']=='absolute' and s['camera_order']==['head','left_wrist','right_wrist']


def test_missing_camera_rejected(tmp_path):
    from ge_act.data.robofollow_schema import read_episode
    p=episode(tmp_path/'scene1'/'task'/'data'/'episode0.hdf5')
    with h5py.File(p,'a') as f:del f['observation/left_camera']
    with pytest.raises((ValueError,KeyError)):read_episode(p)


def test_loader_returns_generic_batch_and_repeat_history(tmp_path):
    from autoresearch.preflight import build_manifest,compute_statistics
    from ge_act.data.robofollow_hdf5_dataset import RoboFollowHDF5Dataset
    episode(tmp_path/'scene1'/'task'/'data'/'episode0.hdf5',n=60)
    mp=tmp_path/'manifest.json';sp=tmp_path/'stats.json'
    m=build_manifest(tmp_path,mp)
    m['episodes'][0]['split']='train';mp.write_text(json.dumps(m));compute_statistics(mp,sp)
    d=RoboFollowHDF5Dataset(mp,sp,fix_sidx=0,sample_size=[32,32])
    b=d[0]
    assert b['video'].shape==(3,3,13,32,32)
    assert b['actions'].shape==(58,14) and b['state'].shape==(1,14)
    assert b['caption']=='pick yellow'
    assert np.isfinite(b['actions'].numpy()).all()
    assert d.frame_indices(0,59)[0][:4]==[0,0,0,0]

def test_history_matches_fifty_action_rollout_stride():
    from ge_act.data.robofollow_hdf5_dataset import RoboFollowHDF5Dataset
    assert RoboFollowHDF5Dataset.frame_indices(200,400)[0][:4]==[50,100,150,200]

def test_loader_rejects_stats_from_another_manifest(tmp_path):
    from autoresearch.preflight import build_manifest,compute_statistics
    from ge_act.data.robofollow_hdf5_dataset import RoboFollowHDF5Dataset
    episode(tmp_path/'scene1'/'task'/'data'/'episode0.hdf5',n=60)
    mp=tmp_path/'manifest.json';sp=tmp_path/'stats.json'
    m=build_manifest(tmp_path,mp);m['episodes'][0]['split']='train';mp.write_text(json.dumps(m));compute_statistics(mp,sp)
    m['seed']=123;mp.write_text(json.dumps(m))
    with pytest.raises(ValueError):RoboFollowHDF5Dataset(mp,sp)

def test_same_trajectory_with_other_instruction_cannot_cross_split(tmp_path):
    from autoresearch.preflight import build_manifest
    # These instruction groups would ordinarily hash to different partitions.
    episode(tmp_path/'scene1'/'task'/'data'/'episode0.hdf5',instruction='pick yellow')
    episode(tmp_path/'scene1'/'other'/'data'/'episode1.hdf5',instruction='pick blue')
    m=build_manifest(tmp_path,tmp_path/'manifest.json')
    assert len(m['episodes'])==2 and len({r['split'] for r in m['episodes']})==1
