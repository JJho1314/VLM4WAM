"""Exercise production loader construction and epoch rollover without loading models."""
import ast
import gc
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate import Accelerator
from test_robofollow_epoch import EpochProbe


@pytest.mark.skipif(not Path('/proc/self/fd').exists(), reason='Linux FD accounting')
def test_epoch_rollover_does_not_leak_file_descriptors(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'ge_act/runner/ge_trainer.py'
    tree = ast.parse(path.read_text())
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
    construction = next(n for n in ast.walk(trainer) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Attribute) and t.attr == 'train_dataloader' for t in n.targets))
    train = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    loop = next(n for n in train.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'set_dataloader_epoch')
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.For))
    names = {'set_dataloader_epoch', '_dataset_children', '_set_wrapped_dataset_epoch'}
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    dataset = EpochProbe()
    state = SimpleNamespace(train_dataset=dataset, train_sampler=None,
                            train_dataloader_generator=torch.Generator(),
                            args=SimpleNamespace(batch_size=2, dataloader_num_workers=2, pin_memory=True))
    namespace = {'torch': torch, 'Any': Any, 'self': state}
    exec(compile(ast.Module(body=helpers + [construction], type_ignores=[]), str(path), 'exec'), namespace)
    # Keep the real workers, queues, pinning thread and atexit registrations;
    # only the tensor pin operation is replaced so this regression runs on CPU.
    state.train_dataloader.pin_memory_device = 'cpu'
    monkeypatch.setattr(torch.utils.data._utils.pin_memory, 'pin_memory', lambda data, device=None: data)
    accelerator = Accelerator(cpu=True)
    state.train_dataloader = accelerator.prepare(state.train_dataloader)
    namespace.update(accelerator=accelerator, cursor=SimpleNamespace(sampler_seed=42, consumed_microbatches=1), first_epoch=0)
    setup = compile(ast.Module(body=loop.body[start:end], type_ignores=[]), str(path), 'exec')
    counts = []
    for epoch in range(24):
        namespace['epoch'] = epoch
        exec(setup, namespace)
        observed = [batch.tolist() for batch in namespace['epoch_dataloader']]
        assert observed == [[epoch, epoch]] * (1 if epoch == 0 else 2)
        gc.collect()
        counts.append(len(os.listdir('/proc/self/fd')))
    print('FD counts across epochs:', counts)
    assert max(counts[4:]) - min(counts[4:]) <= 8, counts
