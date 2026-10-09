import torch
import pytest
from types import SimpleNamespace

class Probe:
    def predict(self,images,instructions):
        assert len(instructions)==len(images)
        assert instructions==['left']*2+['right']*2
        torch.testing.assert_close(images[:,:,0,0,0],torch.tensor([[11,22],[33,33],[44,55],[66,66]],dtype=torch.uint8))
        v=images[:,:,0,0,0].float()
        return SimpleNamespace(tokens=v[:,:,None,None,None].expand(-1,-1,4,256,1024))

def test_three_views_independent_and_metadata():
    from qwen35_baton.robofollow_provider import RoboFollowResearchPlanner,validate_metadata
    images=torch.zeros((2,3,8,8,3),dtype=torch.uint8)
    images[:,0]=11;images[:,1]=22;images[:,2]=33
    images[1,0]=44;images[1,1]=55;images[1,2]=66
    p=RoboFollowResearchPlanner(Probe(),source_checkpoint='probe')
    result=p.predict_tokens(images,['left','right'])
    assert result.shape==(2,3,4,256,1024)
    torch.testing.assert_close(result[0,:,0,0,0],torch.tensor([11.,22.,33.]))
    torch.testing.assert_close(result[1,:,0,0,0],torch.tensor([44.,55.,66.]))
    validate_metadata(p.metadata)
    wrong=dict(p.metadata,camera_order=['left_wrist','head','right_wrist'])
    with pytest.raises(ValueError):validate_metadata(wrong)
    with pytest.raises(ValueError):p.predict_tokens(images[:,:2],['left','right'])


def test_non_uint8_and_instruction_count_rejected():
    from qwen35_baton.robofollow_provider import RoboFollowResearchPlanner
    p=RoboFollowResearchPlanner(Probe(),source_checkpoint='probe')
    with pytest.raises(ValueError):p.predict_tokens(torch.zeros(1,3,8,8,3),['x'])
    with pytest.raises(ValueError):p.predict_tokens(torch.zeros(1,3,8,8,3,dtype=torch.uint8),[])

def test_mixing_extremes_and_joint_gradients():
    from qwen35_baton.robofollow_provider import mix_training_plans
    pred=torch.ones(1,3,4,2,4,requires_grad=True);teacher=torch.zeros_like(pred)
    mixed,loss=mix_training_plans(pred,teacher,1.,.5)
    assert torch.equal(mixed,pred) and loss>0
    (mixed.mean()+loss).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0
    selected,_=mix_training_plans(pred.detach(),teacher,0.,0.)
    assert torch.equal(selected,teacher)
    with pytest.raises(ValueError):mix_training_plans(pred,None,.5,0.)

def test_text_runner_without_semantic_arguments(tmp_path,monkeypatch):
    from pathlib import Path
    import importlib.util,sys,types
    class Base:
        def prepare_models(self):pass
    fake=types.ModuleType('runner.ge_trainer');fake.Trainer=Base
    monkeypatch.setitem(sys.modules,'runner.ge_trainer',fake)
    path=Path(__file__).resolve().parents[1]/'ge_act/runner/robofollow_trainer.py'
    spec=importlib.util.spec_from_file_location('rf_runner_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    runner=object.__new__(module.RoboFollowTrainer)
    runner.args=SimpleNamespace(load_weights=False)
    runner.raw_config={}
    runner.vae=SimpleNamespace(config=SimpleNamespace(latent_channels=128))
    runner.prepare_models()
    assert runner.args.semantic_plan=={}
    assert runner.args.load_diffusion_model_weights is True
