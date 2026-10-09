"""Execute the trainer's actual epoch setup with a tiny CPU dataset."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate import Accelerator
from ge_act.data.robofollow_hdf5_dataset import RoboFollowHDF5Dataset


class EpochProbe(RoboFollowHDF5Dataset):
    def __init__(self):
        self.records = [0, 1, 2, 3]
        self.epoch = 0

    def __getitem__(self, index):
        return self.epoch


@pytest.mark.parametrize('workers', [0, 2])
def test_trainer_epoch_setup_survives_accelerate_loader_recreation(workers):
    # Load only the production setup statements, avoiding model construction and CUDA.
    path = Path(__file__).resolve().parents[1] / 'ge_act/runner/ge_trainer.py'
    tree = ast.parse(path.read_text())
    names = {'set_dataloader_epoch', '_dataset_children', '_set_wrapped_dataset_epoch'}
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    train = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    loop = next(n for n in train.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'set_dataloader_epoch')
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.For))
    namespace = {'torch': torch, 'Any': Any}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), 'exec'), namespace)
    setup = compile(ast.Module(body=loop.body[start:end], type_ignores=[]), str(path), 'exec')
    accelerator = Accelerator(cpu=True)
    dataset = EpochProbe()
    loader = accelerator.prepare(torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=workers, persistent_workers=workers > 0))
    namespace.update(self=SimpleNamespace(train_dataloader=loader), accelerator=accelerator,
                     cursor=SimpleNamespace(sampler_seed=42, consumed_microbatches=0), first_epoch=0)
    for epoch in (0, 1, 2):
        namespace['epoch'] = epoch
        exec(setup, namespace)
        observed = [batch.tolist() for batch in namespace['epoch_dataloader']]
        assert observed == [[epoch, epoch], [epoch, epoch]]
