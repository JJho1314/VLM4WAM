import torch
import pytest
from safetensors.torch import save_file

def test_loading_requires_explicit_new_keys(tmp_path):
    from ge_act.experiments.robofollow_loading import load_robofollow_weights
    m=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Linear(2,14))
    ck=tmp_path/'model.safetensors';save_file({'0.weight':m[0].weight,'0.bias':m[0].bias},str(ck))
    with pytest.raises(ValueError,match='missing'):load_robofollow_weights(m,ck,set())
    report=load_robofollow_weights(m,ck,{'1.weight','1.bias'})
    assert report['new']==['1.bias','1.weight'] and len(report['loaded'])==2

def test_shape_mismatch_and_unexpected_rejected(tmp_path):
    from ge_act.experiments.robofollow_loading import load_robofollow_weights
    m=torch.nn.Linear(3,14);ck=tmp_path/'model.safetensors'
    save_file({'weight':torch.zeros(7,3),'bias':torch.zeros(14)},str(ck))
    with pytest.raises(ValueError,match='shape'):load_robofollow_weights(m,ck,set())
    save_file({'weight':m.weight,'bias':m.bias,'bogus':torch.ones(1)},str(ck))
    with pytest.raises(ValueError,match='unexpected'):load_robofollow_weights(m,ck,set())

def test_ltx_latent_channel_compatibility():
    from types import SimpleNamespace
    from ge_act.experiments.robofollow_loading import ensure_vae_channels
    vae=SimpleNamespace(config=SimpleNamespace(latent_channels=128))
    ensure_vae_channels(vae)
    assert vae.z_dim==128
    with pytest.raises(ValueError):ensure_vae_channels(SimpleNamespace(config=SimpleNamespace(latent_channels=0)))

@pytest.mark.parametrize('field',['camera_order','joint_order','action_type','statistics_sha256','manifest_sha256','planner'])
def test_same_shape_wrong_semantics_rejected(tmp_path,field):
    import json
    from ge_act.experiments.robofollow_loading import write_contract,validate_contract
    ck=tmp_path/'m.safetensors';save_file(torch.nn.Linear(3,14).state_dict(),str(ck))
    expected={'camera_order':['head','left_wrist','right_wrist'],'joint_order':list(range(14)),'action_type':'absolute','statistics_sha256':'s','manifest_sha256':'m','planner':None}
    write_contract(ck,expected,{'capture_phase':'test'})
    validate_contract(ck,expected)
    wrong=dict(expected);wrong[field]='wrong'
    with pytest.raises(ValueError,match='semantic'):validate_contract(ck,wrong)
    ck.write_bytes(b'changed')
    with pytest.raises(ValueError,match='weights'):validate_contract(ck,expected)
