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
