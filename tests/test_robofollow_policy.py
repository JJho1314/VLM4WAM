import numpy as np
import pytest
from ge_act.data.robofollow_schema import CAMERA_ORDER,JOINT_ORDER

def stats():
    return dict(action_type='absolute',action_space='joint',camera_order=CAMERA_ORDER,joint_order=JOINT_ORDER,state_mean=[0]*14,state_std=[1]*14,action_mean=[2]*14,action_std=[3]*14)

def obs():
    return {'observation':{c:{'rgb':np.full((8,8,3),v,np.uint8)} for c,v in zip(('head_camera','left_camera','right_camera'),(1,2,3))},'joint_action':{'vector':np.arange(14,dtype='f4')}}

def test_output_mapping_history_and_instruction_reset():
    from ge_act.experiments.robofollow_policy import RoboFollowPolicy
    seen=[]
    def infer(images,state,instruction):
        seen.append((images.copy(),state.copy(),instruction));return np.zeros((54,14),np.float32)
    p=RoboFollowPolicy(infer,stats());p.set_instruction('left')
    a=p.predict(obs());assert a.shape==(50,14) and a.dtype==np.float32
    np.testing.assert_array_equal(a,2)
    assert seen[0][0].shape==(4,3,8,8,3)
    np.testing.assert_array_equal(seen[0][0][0,:,0,0,0],[1,2,3])
    p.set_instruction('right');assert not p.history
    p.predict(obs());assert seen[-1][2]=='right'
    p.reset();assert not p.history and p.instruction is None

def test_invalid_inputs_and_outputs_fail():
    from ge_act.experiments.robofollow_policy import RoboFollowPolicy
    p=RoboFollowPolicy(lambda *args:np.full((1,14),np.nan),stats());p.set_instruction('x')
    with pytest.raises(ValueError):p.predict(obs())
    bad=obs();del bad['observation']['left_camera']
    with pytest.raises(ValueError):p.predict(bad)
    p=RoboFollowPolicy(lambda *args:np.zeros((1,7)),stats());p.set_instruction('x')
    with pytest.raises(ValueError):p.predict(obs())
    p.reset()
    with pytest.raises(ValueError):p.predict(obs())

def test_state_batch_matches_action_transformer_contract():
    from ge_act.experiments.robofollow_policy import pipeline_state
    assert pipeline_state(np.zeros((1,14),np.float32)).shape==(1,1,14)
    with pytest.raises(ValueError):pipeline_state(np.zeros((14,),np.float32))
