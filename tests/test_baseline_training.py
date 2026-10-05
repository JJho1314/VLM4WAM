import importlib.util
import json
from pathlib import Path
import sys
import types
import torch
from types import SimpleNamespace


def test_no_deadline_still_records_usage(tmp_path):
    from autoresearch.controller import run_process
    from autoresearch.registry import Registry
    record = run_process(['sleep', '.1'], tmp_path/'train', Registry(tmp_path, float('inf')), None, 2)
    assert record['status'] == 'completed' and record['timeout_seconds'] is None
    assert record['gpu_hours'] > 0


def test_fp32_selected_optimizer_parameters(tmp_path, monkeypatch):
    class Base:
        def prepare_optimizer(self):
            self.diffusion_model.frozen.requires_grad_(False)
            self.optimizer = torch.optim.AdamW([self.diffusion_model.action_weight], lr=.01)
    fake = types.ModuleType('runner.ge_trainer'); fake.Trainer = Base
    monkeypatch.setitem(sys.modules, 'runner.ge_trainer', fake)
    path = Path(__file__).resolve().parents[1]/'ge_act/runner/robofollow_trainer.py'
    spec = importlib.util.spec_from_file_location('fp32_runner_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    runner = object.__new__(module.RoboFollowTrainer)
    runner.diffusion_model = torch.nn.Module()
    runner.diffusion_model.action_weight = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    runner.diffusion_model.frozen = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    runner.raw_config = {'robofollow_fp32_trainable': True}
    runner.state = SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True))
    runner.save_folder = str(tmp_path)
    runner.prepare_optimizer()
    p = runner.diffusion_model.action_weight
    assert p.dtype == torch.float32 and runner.diffusion_model.frozen.dtype == torch.bfloat16
    p.sum().backward(); runner.optimizer.step()
    assert not torch.equal(p.detach(), torch.ones(2))
    assert runner.optimizer.state[p]['exp_avg'].dtype == torch.float32


from ge_act.data.robofollow_hdf5_dataset import RoboFollowHDF5Dataset
class EpochDataset(RoboFollowHDF5Dataset):
    def __getitem__(self, index): return self.epoch


def test_persistent_workers_receive_epoch(tmp_path):
    from test_robofollow_data import episode
    from autoresearch.preflight import build_manifest, compute_statistics
    episode(tmp_path/'scene1/task/data/episode0.hdf5', n=60)
    mp, sp = tmp_path/'manifest.json', tmp_path/'stats.json'
    manifest = build_manifest(tmp_path, mp)
    manifest['episodes'][0]['split'] = 'train'
    mp.write_text(json.dumps(manifest)); compute_statistics(mp, sp)
    dataset = EpochDataset(mp, sp)
    loader = torch.utils.data.DataLoader(dataset, num_workers=1, persistent_workers=True)
    assert next(iter(loader)).item() == 0
    dataset.set_epoch(7)
    assert next(iter(loader)).item() == 7
