from types import SimpleNamespace
import logging

import torch

from runner import ge_trainer
from runner.ge_trainer import State, Trainer


def test_non_joint_optimizer_supports_cosine_with_min_lr(monkeypatch):
    monkeypatch.setattr(ge_trainer, "logger", logging.getLogger(__name__))
    trainer = Trainer.__new__(Trainer)
    trainer.args = SimpleNamespace(
        joint_training={},
        train_epochs=1,
        train_steps=10,
        train_mode="all",
        mixed_precision="bf16",
        scale_lr=False,
        lr=2.0e-5,
        semantic_lr=1.0e-4,
        optimizer="adamw",
        beta1=0.9,
        beta2=0.95,
        beta3=0.999,
        epsilon=1.0e-8,
        weight_decay=1.0e-8,
        optimizer_8bit=False,
        optimizer_torchao=False,
        gradient_accumulation_steps=1,
        batch_size=1,
        lr_scheduler="cosine_with_min_lr",
        lr_warmup_steps=1,
        lr_min=5.0e-7,
        lr_num_cycles=1,
        lr_power=1.0,
    )
    trainer.diffusion_model = torch.nn.Linear(2, 2)
    trainer.train_dataloader = [None] * 4
    trainer.state = State()
    trainer.state.accelerator = SimpleNamespace(num_processes=1)

    trainer.prepare_optimizer()

    assert isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.LambdaLR)
