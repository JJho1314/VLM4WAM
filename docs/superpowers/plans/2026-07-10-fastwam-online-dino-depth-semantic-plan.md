# FastWAM Online DINO-Depth Semantic Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a FastWAM-aligned 4B planner and use it online, frozen, in FastWAM Cosmos so that same-position DINO and depth features are fused into a `[B, 1280, 1024]` semantic plan with correct five-keyframe timing and effective-video-FPS-aware RoPE.

**Architecture:** The 4B planner observes the same current multi-camera image and raw instruction as FastWAM, performs one frozen VLM forward, and emits separate DINO and depth branches for five future keyframes. A trainable fusion module owned by the Cosmos video expert normalizes and projects each branch, applies a learnable depth gate, and sends the fused tensor through the existing Cosmos semantic adapter and cross-attention path. The dataset owns the sampled-video FPS contract, while FastWAM routes that FPS and normalized keyframe times through training, standalone inference, and AGRA foresight.

**Tech Stack:** Python 3, PyTorch, Hugging Face Transformers/Qwen3-VL, Hydra/OmegaConf, pytest, FastWAM Cosmos/Wan VAE, shell launchers.

## Global Constraints

- Preserve the FastWAM LIBERO default horizon: 33 raw records sampled at `action_video_freq_ratio=4` produce 9 RGB frames, with the first frame as context and 8 future RGB frames as targets.
- Train the new planner for `sequence_length=9`. Do not reuse the existing 49-frame planner checkpoint as the production provider.
- Use exactly five future semantic keyframes at RGB offsets `[1, 3, 4, 6, 8]`. Their normalized times are `[0.125, 0.375, 0.5, 0.75, 1.0]`.
- Preserve keyframe-major, then row-major spatial ordering. Each branch is `[B, 5 * 16 * 16, 1024] = [B, 1280, 1024]`.
- Run the planner online for both FastWAM training and inference. Keep the VLM, DINO head, depth head, and plan-token embeddings frozen, in evaluation mode, under `torch.no_grad()`, and detached from the FastWAM graph.
- Keep the DINO and depth branches separate until the trainable FastWAM fusion module. Do not concatenate them on the token axis or feature axis.
- The fusion module belongs to the video expert, is included in FastWAM checkpoints, and is the only trainable part of the planner-to-semantic bridge before the existing semantic adapter.
- Keep online and file-backed semantic-plan modes mutually exclusive. Missing depth weights, incompatible metadata, missing FPS, malformed tensor shapes, or non-finite values must fail immediately with a descriptive error.
- Pass the sampled-video FPS, not the raw dataset FPS, to Cosmos: `raw_fps / (global_sample_stride * action_video_freq_ratio)`.
- Preserve all unrelated user changes. Stage and commit only paths explicitly named by each task.

---

## File and Responsibility Map

### Planner training and export

- Modify `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`
  - Centralize planner prompt construction.
  - Add a one-forward dual-output prediction API.
  - Save depth-head weights and complete temporal/tensor metadata.
- Add `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_depth_plan_provider.py`
  - Load and validate a complete frozen planner checkpoint.
  - Convert online image tensors and instructions into Qwen inputs.
  - Return DINO, depth, and normalized keyframe times.
- Add `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k5.sh`
  - Pin the FastWAM-aligned nine-frame, five-keyframe training contract.

### FastWAM model and data integration

- Add `third_party/FastWAM/src/fastwam/models/cosmos/semantic_plan_fusion.py`
  - Implement same-position gated dual-branch fusion.
- Modify `third_party/FastWAM/src/fastwam/models/cosmos/video_expert.py`
  - Own the fusion module and expose a validated fusion method.
- Add `third_party/FastWAM/src/fastwam/models/cosmos/online_semantic_planner.py`
  - Import and construct the external frozen provider without registering it as a trainable child module.
- Modify `third_party/FastWAM/src/fastwam/models/cosmos/runtime.py`
  - Build the fusion module and optional online provider from Hydra configuration.
- Modify `third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py`
  - Invoke the provider from raw video before VAE encoding.
  - Enforce online/file-backed exclusivity.
  - Store semantic times and sampled-video FPS for the current forward.
  - Support online inference from `input_image + prompt`.
- Modify `third_party/FastWAM/src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
  - Preserve the raw LeRobot FPS.
- Modify `third_party/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py`
  - Emit effective sampled-video FPS plus the raw task instruction.
- Modify `third_party/FastWAM/src/fastwam/models/cosmos/couplings/mot.py`
- Modify `third_party/FastWAM/src/fastwam/models/cosmos/couplings/cross_attn.py`
- Modify `third_party/FastWAM/src/fastwam/models/cosmos/couplings/agra.py`
  - Route effective FPS into every Cosmos video-expert call.
- Modify `third_party/FastWAM/configs/model/fastwam_cosmos.yaml`
  - Replace stale semantic geometry and add online-provider/fusion configuration.
- Modify `third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml`
  - Declare the online mode and remove ambiguity with file-backed plans.
- Add `third_party/FastWAM/configs/task/libero_cosmos_2cam224_online_dino_depth.yaml`
  - Enable the online source with an environment-supplied planner checkpoint.

### Tests and runnable verification

- Add `tests/test_lingbot_dino_depth_contract.py`
  - Cover one-forward dual output, checkpoint completeness, metadata, and the nine-frame launcher.
- Add `tests/test_dino_depth_plan_provider.py`
  - Cover metadata rejection, image/prompt preprocessing, output shape, detach, and single invocation.
- Add `tests/test_fastwam_dino_depth_fusion.py`
  - Cover same-position fusion, gate initialization, gradients, validation, and video-expert ownership.
- Modify `tests/test_fastwam_cosmos_semantic_plan.py`
  - Cover dataset FPS, online training/inference routing, exclusivity, and coupling FPS propagation.
- Add `third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py`
  - Provide a checkpoint-backed one-batch runtime smoke test.

---

## Task 1: Make the 4B Planner Contract Dual-Branch and Export-Complete

**Files:**

- Create: `tests/test_lingbot_dino_depth_contract.py`
- Modify: `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`

- [ ] **Step 1: Add failing unit tests for a one-forward DINO+depth API**

Create the test module with lightweight fake components so the test never loads Qwen weights:

```python
from __future__ import annotations

import importlib.util
import sys
import json
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = (
    ROOT
    / 'scripts/qwen3_vl_semantic_planner'
    / 'train_qwen3vl4b_lingbot_dino_planner.py'
)


def load_trainer_module():
    spec = importlib.util.spec_from_file_location('lingbot_planner_trainer', TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CountingHead(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.calls = 0
        self.anchor = nn.Parameter(torch.tensor(value), requires_grad=False)

    def forward(self, image_hidden, plan_hidden):
        self.calls += 1
        batch = plan_hidden.shape[0]
        return torch.full(
            (batch, 5 * 16 * 16, 1024),
            self.value,
            device=plan_hidden.device,
        )


def test_predict_dino_depth_plan_uses_one_vlm_forward():
    module = load_trainer_module()
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.plan_head = CountingHead(1.0)
    wrapper.depth_head = CountingHead(2.0)
    wrapper.plan_head_type = 'lingbot_dino'
    wrapper.vlm_forward_calls = 0

    def fake_forward_hiddens(**_inputs):
        wrapper.vlm_forward_calls += 1
        return torch.zeros(2, 6, 64), torch.zeros(2, 40, 64)

    wrapper._forward_hiddens = fake_forward_hiddens
    dino, depth = wrapper.predict_dino_depth_plan(input_ids=torch.ones(2, 4))

    assert wrapper.vlm_forward_calls == 1
    assert wrapper.plan_head.calls == 1
    assert wrapper.depth_head.calls == 1
    assert dino.shape == depth.shape == (2, 1280, 1024)
    assert torch.all(dino == 1)
    assert torch.all(depth == 2)


def test_predict_dino_depth_plan_requires_depth_head():
    module = load_trainer_module()
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.plan_head = CountingHead(1.0)
    wrapper.depth_head = None
    wrapper.plan_head_type = 'lingbot_dino'
    wrapper._forward_hiddens = lambda **_: (
        torch.zeros(1, 6, 64),
        torch.zeros(1, 40, 64),
    )

    try:
        wrapper.predict_dino_depth_plan(input_ids=torch.ones(1, 4))
    except RuntimeError as error:
        assert 'depth head' in str(error).lower()
    else:
        raise AssertionError('missing depth head must fail')
```

- [ ] **Step 2: Run the focused test and confirm the API is missing**

Run:

```bash
pytest -q tests/test_lingbot_dino_depth_contract.py -k predict_dino_depth
```

Expected: FAIL because `PlannerWrapper.predict_dino_depth_plan` does not exist.

- [ ] **Step 3: Add a shared input builder and the one-forward dual-output method**

In the trainer module, define the prompt template and reusable input builder near `Collator`:

```python
PLANNER_USER_TEMPLATE = (
    'You are a robot video semantic planner. Given the first frame and instruction, '
    'predict future spatial semantic plan tokens for the manipulation video.\n'
    'Instruction: {instruction}'
)


def build_planner_inputs(processor, images, instructions, plan_sequence):
    if len(images) != len(instructions):
        raise ValueError(
            f'images/instructions batch mismatch: {len(images)} != {len(instructions)}'
        )
    plan_text = (
        plan_sequence
        if isinstance(plan_sequence, str)
        else ' '.join(plan_sequence)
    )
    conversations = []
    for image, instruction in zip(images, instructions, strict=True):
        conversations.append(
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image'},
                        {
                            'type': 'text',
                            'text': PLANNER_USER_TEMPLATE.format(
                                instruction=str(instruction)
                            ),
                        },
                    ],
                },
                {
                    'role': 'assistant',
                    'content': plan_text,
                },
            ]
        )
    texts = [
        processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conversation in conversations
    ]
    return processor(
        text=texts,
        images=list(images),
        padding=True,
        return_tensors='pt',
    )
```

Change `Collator.__call__` to collect its images and raw prompts, then call this function. Do not maintain a second copy of the user prompt.

Add this method beside `predict_semantic_plan`:

```python
def predict_dino_depth_plan(self, **model_inputs):
    if self.plan_head_type != 'lingbot_dino':
        raise RuntimeError(
            'DINO+depth prediction requires plan_head_type=lingbot_dino'
        )
    if self.depth_head is None:
        raise RuntimeError(
            'DINO+depth prediction requires a configured depth head'
        )
    image_hidden, plan_hidden = self._forward_hiddens(**model_inputs)
    dino_dtype = next(self.plan_head.parameters()).dtype
    depth_dtype = next(self.depth_head.parameters()).dtype
    dino_plan = self.plan_head(
        image_hidden.to(dtype=dino_dtype),
        plan_hidden.to(dtype=dino_dtype),
    ).float()
    depth_plan = self.depth_head(
        image_hidden.to(dtype=depth_dtype),
        plan_hidden.to(dtype=depth_dtype),
    ).float()
    if dino_plan.shape != depth_plan.shape:
        raise RuntimeError(
            'DINO/depth head output mismatch: '
            f'{tuple(dino_plan.shape)} != {tuple(depth_plan.shape)}'
        )
    return dino_plan, depth_plan
```

Refactor the training `forward` method to call `_forward_hiddens` once and reuse those hidden states for both losses. Do not call `predict_dino_depth_plan` from training if the DINO-only mode must remain supported.

- [ ] **Step 4: Add failing tests for checkpoint completeness and exact metadata**

Append:

```python
class FakeSaveableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 8)

    def get_input_embeddings(self):
        return self.embedding

    def save_pretrained(self, output_dir, **_kwargs):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'config.json').write_text('{}')


class FakeProcessor:
    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'processor_config.json').write_text('{}')


def test_save_checkpoint_writes_depth_and_fastwam_contract(tmp_path):
    module = load_trainer_module()
    wrapper = module.PlannerWrapper.__new__(module.PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = FakeSaveableModel()
    wrapper.plan_head = nn.Linear(8, 8)
    wrapper.depth_head = nn.Linear(8, 8)
    wrapper.plan_token_ids = list(range(3, 43))
    wrapper.plan_head_type = 'lingbot_dino'
    wrapper.num_latent_per_keyframe = 8
    wrapper.latent_len = 40
    wrapper.target_len = 1280
    wrapper.plan_head_num_heads = 16
    wrapper.plan_head_dropout = 0.0
    wrapper.sem_mlp_hidden_size = 0
    wrapper.mse_loss_weight = 1.0
    wrapper.cosine_loss_weight = 0.0
    wrapper.norm_loss_weight = 0.0
    wrapper.variance_loss_weight = 0.0
    wrapper.infonce_loss_weight = 0.0
    wrapper.infonce_temperature = 0.07
    wrapper.depth_loss_weight = 0.004
    args = Namespace(
        sample_feature_type='lingbot_dino_depth',
        plan_label_dir=None,
        sample_one_window_per_stem=False,
        online_plan_labels=True,
        keyframe_scheme='uniform',
        keyframe_gamma=0.6,
        sequence_length=9,
        online_grid_size=16,
        siglip2_encoder_path=None,
        frame_ranges_json=None,
        fastwam_data_config=Path('configs/data/libero_2cam_cosmos.yaml'),
        model_path=Path('Qwen3-VL-4B-lingbot-vlm'),
        num_keyframes=5,
        grid_size=16,
        semantic_dim=1024,
        train_plan_token_embedding=True,
        full_finetune=True,
        freeze_vision=True,
        freeze_lm_head=True,
        use_depth=True,
        depth_dim=1024,
        depth_grid_size=16,
    )

    module.save_checkpoint(
        tmp_path,
        7,
        wrapper,
        FakeProcessor(),
        args,
        rank=0,
    )

    checkpoint = tmp_path / 'step_000007'
    assert (checkpoint / 'qwen3vl_lora_or_model/config.json').is_file()
    assert (checkpoint / 'processor/processor_config.json').is_file()
    assert (checkpoint / 'plan_head.pt').is_file()
    assert (checkpoint / 'depth_head.pt').is_file()
    assert (checkpoint / 'plan_token_embedding.pt').is_file()
    metadata = json.loads((checkpoint / 'planner_meta.json').read_text())
    assert metadata['sequence_length'] == 9
    assert metadata['num_keyframes'] == 5
    assert metadata['grid_size'] == 16
    assert metadata['semantic_dim'] == 1024
    assert metadata['target_tokens'] == 1280
    assert metadata['keyframe_offsets'] == [1, 3, 4, 6, 8]
    assert metadata['has_depth_head'] is True
    assert metadata['token_order'] == 'keyframe_major_row_major'
    assert metadata['plan_token_strings'] == [
        f'<|sem_plan_{index}|>' for index in range(40)
    ]
```

- [ ] **Step 5: Run the checkpoint test and confirm depth export is absent**

Run:

```bash
pytest -q tests/test_lingbot_dino_depth_contract.py -k save_checkpoint
```

Expected: FAIL because `depth_head.pt` and the complete metadata contract are absent.

- [ ] **Step 6: Export both heads and an exact provider contract**

In `save_checkpoint`, unwrap the training module once, save the existing artifacts, then add:

```python
depth_head = getattr(module, 'depth_head', None)
if depth_head is not None:
    torch.save(depth_head.state_dict(), output_dir / 'depth_head.pt')

offsets = keyframe_offsets(
    sequence_length=int(args.sequence_length),
    n=int(args.num_keyframes),
    scheme=str(args.keyframe_scheme),
    gamma=float(args.keyframe_gamma),
)
meta.update(
    {
        'sequence_length': int(args.sequence_length),
        'num_keyframes': int(args.num_keyframes),
        'grid_size': int(args.grid_size),
        'semantic_dim': int(args.semantic_dim),
        'target_tokens': int(module.target_len),
        'keyframe_offsets': [int(offset) for offset in offsets],
        'normalized_keyframe_times': [
            float(offset) / float(args.sequence_length - 1)
            for offset in offsets
        ],
        'has_depth_head': depth_head is not None,
        'depth_feature_dim': (
            int(args.depth_dim) if depth_head is not None else None
        ),
        'depth_grid_size': int(args.depth_grid_size),
        'depth_loss_weight': float(module.depth_loss_weight),
        'num_latent_per_keyframe': int(module.num_latent_per_keyframe),
        'plan_token_strings': [
            f'<|sem_plan_{index}|>' for index in range(module.latent_len)
        ],
        'token_order': 'keyframe_major_row_major',
        'planner_input_frame': (
            'fastwam_current_multicamera_composite'
            if args.fastwam_data_config is not None
            else 'legacy_single_current_frame'
        ),
    }
)
(output_dir / 'planner_meta.json').write_text(
    json.dumps(meta, indent=2, sort_keys=True) + '\n'
)
```

Here `output_dir` in the snippet denotes the per-step `ckpt` directory already created by `save_checkpoint`; use that existing variable name in the implementation.

Keep the temporal geometry in `args` and the head geometry in the existing wrapper attributes; do not introduce a second unsynchronized copy.

- [ ] **Step 7: Run all planner contract tests**

Run:

```bash
pytest -q tests/test_lingbot_dino_depth_contract.py
```

Expected: PASS.

- [ ] **Step 8: Commit only the planner contract changes**

Run:

```bash
git add \
  tests/test_lingbot_dino_depth_contract.py \
  scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py
git commit -m 'feat: export dual-branch semantic planner'
```

---

## Task 2: Implement the Frozen Online DINO+Depth Provider

**Files:**

- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_depth_plan_provider.py`
- Create: `tests/test_dino_depth_plan_provider.py`
- Modify: `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py` only if an import-safe helper needs a small correction

- [ ] **Step 1: Add failing metadata validation tests**

Create:

```python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_PATH = (
    ROOT
    / 'scripts/qwen3_vl_semantic_planner/lingbot_dino_4b'
    / 'dino_depth_plan_provider.py'
)


def load_provider_module():
    spec = importlib.util.spec_from_file_location('dino_depth_provider', PROVIDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_metadata():
    return {
        'sequence_length': 9,
        'num_keyframes': 5,
        'grid_size': 16,
        'semantic_dim': 1024,
        'target_tokens': 1280,
        'keyframe_offsets': [1, 3, 4, 6, 8],
        'normalized_keyframe_times': [0.125, 0.375, 0.5, 0.75, 1.0],
        'has_depth_head': True,
        'depth_feature_dim': 1024,
        'depth_grid_size': 16,
        'num_latent_per_keyframe': 8,
        'plan_head_type': 'lingbot_dino',
        'planner_input_frame': 'fastwam_current_multicamera_composite',
        'plan_token_strings': [
            f'<|sem_plan_{index}|>' for index in range(40)
        ],
        'token_order': 'keyframe_major_row_major',
    }


def test_validate_metadata_accepts_exact_fastwam_contract():
    module = load_provider_module()
    contract = module.validate_planner_metadata(valid_metadata())
    assert contract.keyframe_offsets == (1, 3, 4, 6, 8)
    assert contract.normalized_keyframe_times == (
        0.125,
        0.375,
        0.5,
        0.75,
        1.0,
    )


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('sequence_length', 49),
        ('num_keyframes', 6),
        ('grid_size', 9),
        ('semantic_dim', 1152),
        ('target_tokens', 486),
        ('keyframe_offsets', [1, 2, 3, 4, 8]),
        ('has_depth_head', False),
        ('num_latent_per_keyframe', 4),
        ('planner_input_frame', 'legacy_single_current_frame'),
        ('token_order', 'spatial_major'),
    ],
)
def test_validate_metadata_rejects_incompatible_checkpoint(field, value):
    module = load_provider_module()
    metadata = valid_metadata()
    metadata[field] = value
    with pytest.raises(ValueError, match=field):
        module.validate_planner_metadata(metadata)
```

- [ ] **Step 2: Run the validation tests and confirm the provider is absent**

Run:

```bash
pytest -q tests/test_dino_depth_plan_provider.py -k metadata
```

Expected: FAIL because the provider module does not exist.

- [ ] **Step 3: Implement the immutable planner contract and file checks**

Start the provider with:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


EXPECTED_METADATA = {
    'sequence_length': 9,
    'num_keyframes': 5,
    'grid_size': 16,
    'semantic_dim': 1024,
    'target_tokens': 1280,
    'keyframe_offsets': [1, 3, 4, 6, 8],
    'has_depth_head': True,
    'depth_feature_dim': 1024,
    'depth_grid_size': 16,
    'num_latent_per_keyframe': 8,
    'plan_head_type': 'lingbot_dino',
    'planner_input_frame': 'fastwam_current_multicamera_composite',
    'token_order': 'keyframe_major_row_major',
}


@dataclass(frozen=True)
class PlannerContract:
    sequence_length: int
    num_keyframes: int
    grid_size: int
    semantic_dim: int
    target_tokens: int
    keyframe_offsets: tuple[int, ...]
    normalized_keyframe_times: tuple[float, ...]
    plan_token_strings: tuple[str, ...]


@dataclass(frozen=True)
class DinoDepthPlan:
    dino_plan: torch.Tensor
    depth_plan: torch.Tensor
    semantic_plan_times: torch.Tensor


def validate_planner_metadata(metadata: dict) -> PlannerContract:
    for field, expected in EXPECTED_METADATA.items():
        actual = metadata.get(field)
        if actual != expected:
            raise ValueError(
                f'incompatible planner metadata field {field}: '
                f'expected {expected!r}, got {actual!r}'
            )
    expected_times = tuple(
        offset / (EXPECTED_METADATA['sequence_length'] - 1)
        for offset in EXPECTED_METADATA['keyframe_offsets']
    )
    actual_times = tuple(float(value) for value in metadata.get(
        'normalized_keyframe_times', ()
    ))
    if len(actual_times) != len(expected_times) or any(
        abs(actual - expected) > 1e-7
        for actual, expected in zip(actual_times, expected_times, strict=True)
    ):
        raise ValueError(
            'incompatible planner metadata field normalized_keyframe_times: '
            f'expected {expected_times!r}, got {actual_times!r}'
        )
    expected_plan_tokens = tuple(
        f'<|sem_plan_{index}|>' for index in range(5 * 8)
    )
    actual_plan_tokens = tuple(metadata.get('plan_token_strings', ()))
    if actual_plan_tokens != expected_plan_tokens:
        raise ValueError(
            'incompatible planner metadata field plan_token_strings: '
            f'expected {expected_plan_tokens!r}, got {actual_plan_tokens!r}'
        )
    return PlannerContract(
        sequence_length=9,
        num_keyframes=5,
        grid_size=16,
        semantic_dim=1024,
        target_tokens=1280,
        keyframe_offsets=(1, 3, 4, 6, 8),
        normalized_keyframe_times=expected_times,
        plan_token_strings=expected_plan_tokens,
    )


def validate_checkpoint_files(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    required_files = (
        'plan_head.pt',
        'depth_head.pt',
        'plan_token_embedding.pt',
        'planner_meta.json',
    )
    missing = [
        name for name in required_files
        if not (checkpoint_dir / name).is_file()
    ]
    required_dirs = ('qwen3vl_lora_or_model', 'processor')
    missing.extend(
        name for name in required_dirs
        if not (checkpoint_dir / name).is_dir()
    )
    if missing:
        raise FileNotFoundError(
            f'incomplete planner checkpoint {checkpoint_dir}: missing {missing}'
        )
    return checkpoint_dir
```

In a separate test, remove each required file or directory in turn and assert that its exact name appears in the error.

- [ ] **Step 4: Add failing online prediction tests with fake processor and wrapper**

Append to the test:

```python
class FakeBatch(dict):
    def to(self, device):
        return FakeBatch(
            {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in self.items()
            }
        )


class FakeProcessor:
    def __init__(self):
        self.images = None
        self.instructions = None

    def build_inputs(self, images, instructions, _plan_sequence):
        self.images = images
        self.instructions = list(instructions)
        return FakeBatch({'input_ids': torch.ones(len(images), 4, dtype=torch.long)})


class FakeWrapper:
    def __init__(self):
        self.calls = 0
        self.training = True

    def eval(self):
        self.training = False
        return self

    def parameters(self):
        return iter(())

    def predict_dino_depth_plan(self, **inputs):
        self.calls += 1
        batch = inputs['input_ids'].shape[0]
        dino = torch.ones(batch, 1280, 1024, requires_grad=True)
        depth = torch.full(
            (batch, 1280, 1024),
            2.0,
            requires_grad=True,
        )
        return dino, depth


def test_predict_returns_detached_dual_branch_and_times():
    module = load_provider_module()
    processor = FakeProcessor()
    wrapper = FakeWrapper()
    provider = module.FrozenDinoDepthPlanProvider.from_components(
        processor=processor,
        wrapper=wrapper,
        contract=module.validate_planner_metadata(valid_metadata()),
        device=torch.device('cpu'),
        input_builder=processor.build_inputs,
    )
    images = torch.zeros(2, 3, 12, 20)
    result = provider.predict(images, ['open drawer', 'pick mug'])

    assert wrapper.calls == 1
    assert wrapper.training is False
    assert result.dino_plan.shape == (2, 1280, 1024)
    assert result.depth_plan.shape == (2, 1280, 1024)
    assert result.semantic_plan_times.shape == (2, 5)
    assert result.dino_plan.requires_grad is False
    assert result.depth_plan.requires_grad is False
    assert processor.instructions == ['open drawer', 'pick mug']
    assert all(isinstance(image, Image.Image) for image in processor.images)
```

- [ ] **Step 5: Run the prediction test and confirm the provider class is missing**

Run:

```bash
pytest -q tests/test_dino_depth_plan_provider.py -k predict_returns
```

Expected: FAIL because `FrozenDinoDepthPlanProvider` is not implemented.

- [ ] **Step 6: Implement tensor conversion, frozen construction, and prediction**

Add:

```python
def image_tensor_batch_to_pil(images: torch.Tensor) -> list[Image.Image]:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f'planner images must have shape [B, 3, H, W], got {tuple(images.shape)}'
        )
    if not torch.isfinite(images).all():
        raise ValueError('planner images contain non-finite values')
    images = images.detach().to(device='cpu', dtype=torch.float32)
    if images.min().item() < -1.0001 or images.max().item() > 1.0001:
        raise ValueError('planner images must be normalized to [-1, 1]')
    images = ((images + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return [
        Image.fromarray(image.permute(1, 2, 0).numpy(), mode='RGB')
        for image in images
    ]


class FrozenDinoDepthPlanProvider:
    def __init__(
        self,
        *,
        processor,
        wrapper,
        contract: PlannerContract,
        device: torch.device,
        input_builder,
        input_mover,
    ):
        self.processor = processor
        self.wrapper = wrapper.eval()
        self.contract = contract
        self.device = torch.device(device)
        self.input_builder = input_builder
        self.input_mover = input_mover
        for parameter in self.wrapper.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_components(
        cls,
        *,
        processor,
        wrapper,
        contract,
        device,
        input_builder,
        input_mover=None,
    ):
        return cls(
            processor=processor,
            wrapper=wrapper,
            contract=contract,
            device=device,
            input_builder=input_builder,
            input_mover=(
                input_mover
                if input_mover is not None
                else lambda inputs: inputs.to(device)
            ),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ):
        checkpoint_dir = validate_checkpoint_files(checkpoint_dir)
        metadata = json.loads(
            (checkpoint_dir / 'planner_meta.json').read_text()
        )
        contract = validate_planner_metadata(metadata)

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from train_qwen3vl4b_lingbot_dino_planner import (
            PlannerWrapper,
            build_planner_inputs,
            move_qwen_inputs_to_device,
        )

        processor = AutoProcessor.from_pretrained(
            checkpoint_dir / 'processor',
            local_files_only=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_dir / 'qwen3vl_lora_or_model',
            torch_dtype=dtype,
            local_files_only=True,
        ).to(device)
        wrapper = PlannerWrapper.from_exported_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            metadata=metadata,
        ).to(device)
        return cls(
            processor=processor,
            wrapper=wrapper,
            contract=contract,
            device=torch.device(device),
            input_builder=lambda images, instructions, plan_sequence: (
                build_planner_inputs(
                    processor,
                    images,
                    instructions,
                    plan_sequence,
                )
            ),
            input_mover=lambda inputs: move_qwen_inputs_to_device(
                inputs,
                torch.device(device),
                model_dtype=dtype,
            ),
        )

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        instructions: Sequence[str],
    ) -> DinoDepthPlan:
        if images.shape[0] != len(instructions):
            raise ValueError(
                f'image/instruction batch mismatch: '
                f'{images.shape[0]} != {len(instructions)}'
            )
        pil_images = image_tensor_batch_to_pil(images)
        model_inputs = self.input_mover(
            self.input_builder(
                pil_images,
                list(instructions),
                list(self.contract.plan_token_strings),
            )
        )
        dino_plan, depth_plan = self.wrapper.predict_dino_depth_plan(
            **model_inputs
        )
        expected_shape = (
            images.shape[0],
            self.contract.target_tokens,
            self.contract.semantic_dim,
        )
        for name, tensor in (
            ('dino_plan', dino_plan),
            ('depth_plan', depth_plan),
        ):
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f'{name} must have shape {expected_shape}, '
                    f'got {tuple(tensor.shape)}'
                )
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f'{name} contains non-finite values')
        times = torch.tensor(
            self.contract.normalized_keyframe_times,
            device=dino_plan.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(images.shape[0], -1)
        return DinoDepthPlan(
            dino_plan=dino_plan.detach(),
            depth_plan=depth_plan.detach(),
            semantic_plan_times=times.detach(),
        )
```

Implement `PlannerWrapper.from_exported_checkpoint` in the trainer module as the single reconstruction path. It must:

1. Instantiate the DINO and depth heads from metadata.
2. Load `plan_head.pt` and `depth_head.pt` with `weights_only=True`.
3. Restore the plan-token rows from `plan_token_embedding.pt`.
4. Use `strict=True` for both head state dictionaries.
5. Return `eval()` with every parameter frozen.

Do not duplicate the head architecture inside the provider.

Use this classmethod:

```python
@classmethod
def from_exported_checkpoint(
    cls,
    *,
    model: nn.Module,
    checkpoint_dir: str | Path,
    metadata: dict[str, Any],
) -> 'PlannerWrapper':
    checkpoint_dir = Path(checkpoint_dir)
    text_config = getattr(model.config, 'text_config', model.config)
    hidden_size = int(text_config.hidden_size)
    wrapper = cls(
        model=model,
        hidden_size=hidden_size,
        semantic_dim=int(metadata['semantic_dim']),
        plan_token_ids=[int(value) for value in metadata['plan_token_ids']],
        target_len=int(metadata['target_tokens']),
        num_keyframes=int(metadata['num_keyframes']),
        grid_size=int(metadata['grid_size']),
        num_latent_per_keyframe=int(metadata['num_latent_per_keyframe']),
        plan_head_type=str(metadata['plan_head_type']),
        plan_head_num_heads=int(metadata['plan_head_num_heads']),
        plan_head_dropout=float(metadata['plan_head_dropout']),
        sem_mlp_hidden_size=int(metadata['sem_mlp_hidden_size']),
        mse_loss_weight=float(metadata['mse_loss_weight']),
        cosine_loss_weight=float(metadata['cosine_loss_weight']),
        norm_loss_weight=float(metadata['norm_loss_weight']),
        variance_loss_weight=float(metadata['variance_loss_weight']),
        infonce_loss_weight=float(metadata['infonce_loss_weight']),
        infonce_temperature=float(metadata['infonce_temperature']),
        use_depth=True,
        depth_dim=int(metadata['depth_feature_dim']),
        depth_grid_size=int(metadata['depth_grid_size']),
        depth_loss_weight=float(metadata['depth_loss_weight']),
    )
    wrapper.plan_head.load_state_dict(
        torch.load(
            checkpoint_dir / 'plan_head.pt',
            map_location='cpu',
            weights_only=True,
        ),
        strict=True,
    )
    wrapper.depth_head.load_state_dict(
        torch.load(
            checkpoint_dir / 'depth_head.pt',
            map_location='cpu',
            weights_only=True,
        ),
        strict=True,
    )
    plan_embedding = torch.load(
        checkpoint_dir / 'plan_token_embedding.pt',
        map_location='cpu',
        weights_only=True,
    )
    plan_ids = torch.tensor(wrapper.plan_token_ids, dtype=torch.long)
    if tuple(plan_embedding.shape) != (
        len(wrapper.plan_token_ids),
        model.get_input_embeddings().weight.shape[1],
    ):
        raise ValueError(
            f'incompatible plan token embedding shape: '
            f'{tuple(plan_embedding.shape)}'
        )
    with torch.no_grad():
        model.get_input_embeddings().weight[plan_ids].copy_(
            plan_embedding.to(model.get_input_embeddings().weight.dtype)
        )
    wrapper.eval()
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    return wrapper
```

Add `depth_loss_weight` to exported metadata so this reconstruction has no hidden default.

- [ ] **Step 7: Add output validation failure tests**

Add tests for:

```python
def test_predict_rejects_mismatched_instruction_count():
    module = load_provider_module()
    processor = FakeProcessor()
    provider = module.FrozenDinoDepthPlanProvider.from_components(
        processor=processor,
        wrapper=FakeWrapper(),
        contract=module.validate_planner_metadata(valid_metadata()),
        device='cpu',
        input_builder=processor.build_inputs,
    )
    with pytest.raises(ValueError, match='batch mismatch'):
        provider.predict(torch.zeros(2, 3, 8, 8), ['only one'])


def test_image_tensor_batch_rejects_non_finite_input():
    module = load_provider_module()
    images = torch.zeros(1, 3, 8, 8)
    images[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        module.image_tensor_batch_to_pil(images)
```

- [ ] **Step 8: Run the provider suite**

Run:

```bash
pytest -q tests/test_dino_depth_plan_provider.py
```

Expected: PASS.

- [ ] **Step 9: Commit only the provider implementation**

Run:

```bash
git add \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/dino_depth_plan_provider.py \
  scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py \
  tests/test_dino_depth_plan_provider.py
git commit -m 'feat: add frozen dino depth plan provider'
```

---

## Task 3: Pin a Nine-Frame, Five-Keyframe DINO+Depth Training Entry Point

**Files:**

- Modify: `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`
- Create: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k5.sh`
- Modify: `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh`
- Modify: `third_party/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py`
- Modify: `tests/test_lingbot_dino_depth_contract.py`

- [ ] **Step 1: Add a failing test that converts the exact FastWAM sample into planner input**

Append:

```python
def test_fastwam_planner_dataset_uses_composed_nine_frame_video():
    module = load_trainer_module()

    class FakeFastWAMDataset:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            video = torch.linspace(
                -1.0,
                1.0,
                steps=3 * 9 * 8 * 16,
            ).reshape(3, 9, 8, 16)
            return {
                'video': video,
                'instruction': 'open the middle drawer',
            }

    dataset = module.FastWAMOnlinePlannerDataset.from_dataset(
        FakeFastWAMDataset(),
        max_samples=0,
    )
    item = dataset[0]

    assert item['image'].size == (16, 8)
    assert item['prompt'] == 'open the middle drawer'
    assert item['keyframe_images'].shape == (5, 8, 16, 3)
    assert item['current_image'].shape == (8, 16, 3)
    assert dataset.offsets == [1, 3, 4, 6, 8]
```

- [ ] **Step 2: Run the adapter test and confirm the dataset class is absent**

Run:

```bash
pytest -q tests/test_lingbot_dino_depth_contract.py \
  -k fastwam_planner_dataset
```

Expected: FAIL because `FastWAMOnlinePlannerDataset` does not exist.

- [ ] **Step 3: Add a planner dataset adapter over the real FastWAM data config**

Add `--fastwam-data-config` to the trainer parser. Make `--dataset-root` optional at parse time, then validate that the FastWAM config and the legacy dataset root are not both selected.

Implement:

```python
class FastWAMOnlinePlannerDataset(Dataset):
    offsets = [1, 3, 4, 6, 8]

    def __init__(self, dataset, max_samples: int = 0):
        self.dataset = dataset
        self.max_samples = int(max_samples)

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        *,
        dataset_dirs: Sequence[str] | None = None,
        max_samples: int = 0,
    ):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        data_config = OmegaConf.load(config_path)
        root_config = OmegaConf.create(
            {
                'data': OmegaConf.to_container(
                    data_config,
                    resolve=False,
                )
            }
        )
        if dataset_dirs:
            root_config.data.train.dataset_dirs = [
                str(path) for path in dataset_dirs
            ]
        dataset = instantiate(root_config.data.train)
        return cls(dataset, max_samples=max_samples)

    @classmethod
    def from_dataset(cls, dataset, max_samples: int = 0):
        return cls(dataset, max_samples=max_samples)

    def __len__(self):
        size = len(self.dataset)
        return min(size, self.max_samples) if self.max_samples > 0 else size

    def set_epoch(self, _epoch: int) -> None:
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        video = sample['video']
        if tuple(video.shape[:2]) != (3, 9):
            raise ValueError(
                'FastWAM planner input must be [3, 9, H, W], got '
                f'{tuple(video.shape)}'
            )
        instruction = sample.get('instruction')
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(
                'FastWAM planner sample needs a non-empty raw instruction'
            )
        selected = video[:, [0, *self.offsets]]
        selected = (
            ((selected.to(torch.float32) + 1.0) * 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .contiguous()
        )
        current = selected[0]
        return {
            'stem': f'fastwam_{index:09d}',
            'sample_id': f'fastwam_{index:09d}',
            'image': Image.fromarray(current.numpy(), mode='RGB'),
            'prompt': instruction,
            'keyframe_images': selected[1:],
            'current_image': current,
        }
```

Add repeatable `--fastwam-dataset-dir` arguments and pass them to `from_config` so machine-local dataset locations can override the relative paths in the shared YAML. When `--online-plan-labels --fastwam-data-config PATH` is selected, construct this adapter instead of `OnlineSemanticPlanDataset`. Require `sequence_length=9`, `num_keyframes=5`, and the exact offsets. The DINO and depth teachers continue to run online over `current_image` and `keyframe_images`.

Add the raw instruction to the existing FastWAM sample dictionary in `robot_video_dataset.py`:

```python
'instruction': str(task),
```

This is the unformatted task string; keep the existing formatted `prompt` for the FastWAM text cache.

- [ ] **Step 4: Run the aligned-dataset test**

Run:

```bash
pytest -q tests/test_lingbot_dino_depth_contract.py \
  -k fastwam_planner_dataset
```

Expected: PASS.

- [ ] **Step 5: Add a failing static contract test for the launcher**

Append:

```python
def test_fastwam_launcher_pins_nine_frame_dual_branch_contract():
    launcher = (
        ROOT
        / 'scripts/qwen3_vl_semantic_planner/lingbot_dino_4b'
        / 'train_lingbot_dino_depth_fastwam_k5.sh'
    ).read_text()
    required_exports = (
        'export USE_DEPTH=1',
        'export SEQUENCE_LENGTH=9',
        'export NUM_KEYFRAMES=5',
        'export GRID_SIZE=16',
        'export SEMANTIC_DIM=1024',
        'export KEYFRAME_SCHEME=uniform',
        'export FASTWAM_DATA_CONFIG=',
    )
    for export in required_exports:
        assert export in launcher
    assert 'train_lingbot_dino_4b.sh' in launcher
```

- [ ] **Step 6: Run the launcher test and confirm the file is missing**

Run:

```bash
pytest -q tests/test_lingbot_dino_depth_contract.py \
  -k fastwam_launcher
```

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 7: Create the explicit FastWAM-aligned wrapper**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

export USE_DEPTH=1
export SEQUENCE_LENGTH=9
export NUM_KEYFRAMES=5
export GRID_SIZE=16
export SEMANTIC_DIM=1024
export KEYFRAME_SCHEME=uniform
export FASTWAM_DATA_CONFIG=${FASTWAM_DATA_CONFIG:-third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3vl4b_lingbot_dino_depth_fastwam_k5}

exec "$SCRIPT_DIR/train_lingbot_dino_4b.sh" "$@"
```

Make it executable:

```bash
chmod +x \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k5.sh
```

Change the base launcher so `DATASET_ROOT` and `FASTWAM_DATA_CONFIG` are mutually exclusive inputs, with at least one required. When `FASTWAM_DATA_CONFIG` is set, append:

```bash
--fastwam-data-config "$FASTWAM_DATA_CONFIG"
```

and do not append `--dataset-root` or `--frame-ranges-json`. If the colon-separated `FASTWAM_DATASET_DIRS` variable is non-empty, split it and append one `--fastwam-dataset-dir` argument per directory. The base launcher must continue to honor environment overrides using `VAR=${VAR:-default}`.

- [ ] **Step 8: Run the contract test and a shell syntax check**

Run:

```bash
pytest -q tests/test_lingbot_dino_depth_contract.py \
  -k fastwam_launcher
bash -n \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k5.sh
```

Expected: both commands PASS.

- [ ] **Step 9: Commit the aligned dataset mode and launcher**

Run:

```bash
git add \
  scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k5.sh \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh \
  third_party/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py \
  tests/test_lingbot_dino_depth_contract.py
git commit -m 'feat: add fastwam aligned planner training entrypoint'
```

---

## Task 4: Add Trainable Same-Position Dual-Branch Fusion

**Files:**

- Create: `third_party/FastWAM/src/fastwam/models/cosmos/semantic_plan_fusion.py`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/video_expert.py`
- Create: `tests/test_fastwam_dino_depth_fusion.py`

- [ ] **Step 1: Add failing fusion behavior and gradient tests**

Create:

```python
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
FUSION_PATH = (
    ROOT
    / 'third_party/FastWAM/src/fastwam/models/cosmos'
    / 'semantic_plan_fusion.py'
)


def load_fusion_module():
    spec = importlib.util.spec_from_file_location('semantic_plan_fusion', FUSION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fusion_preserves_same_position_shape_and_initial_gate():
    module = load_fusion_module()
    fusion = module.DinoDepthPlanFusion(
        feature_dim=1024,
        max_tokens=1280,
        initial_depth_gate=0.1,
    )
    dino = torch.randn(2, 1280, 1024)
    depth = torch.randn(2, 1280, 1024)
    output = fusion(dino, depth)

    assert output.shape == dino.shape
    assert math.isclose(
        torch.sigmoid(fusion.depth_gate_logit).item(),
        0.1,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_fusion_trains_both_projections_and_gate():
    module = load_fusion_module()
    fusion = module.DinoDepthPlanFusion(1024, 1280, 0.1)
    output = fusion(
        torch.randn(1, 1280, 1024),
        torch.randn(1, 1280, 1024),
    )
    output.square().mean().backward()
    assert fusion.dino_proj.weight.grad is not None
    assert fusion.depth_proj.weight.grad is not None
    assert fusion.depth_gate_logit.grad is not None


@pytest.mark.parametrize(
    ('dino_shape', 'depth_shape', 'message'),
    [
        ((1, 1279, 1024), (1, 1279, 1024), '1280'),
        ((1, 1280, 1023), (1, 1280, 1023), '1024'),
        ((1, 1280, 1024), (2, 1280, 1024), 'same shape'),
    ],
)
def test_fusion_rejects_contract_violations(
    dino_shape,
    depth_shape,
    message,
):
    module = load_fusion_module()
    fusion = module.DinoDepthPlanFusion(1024, 1280, 0.1)
    with pytest.raises(ValueError, match=message):
        fusion(torch.zeros(dino_shape), torch.zeros(depth_shape))
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run:

```bash
pytest -q tests/test_fastwam_dino_depth_fusion.py
```

Expected: FAIL because `semantic_plan_fusion.py` does not exist.

- [ ] **Step 3: Implement the exact fusion equation**

Create:

```python
from __future__ import annotations

import math

import torch
from torch import nn


class DinoDepthPlanFusion(nn.Module):
    def __init__(
        self,
        feature_dim: int = 1024,
        max_tokens: int = 1280,
        initial_depth_gate: float = 0.1,
    ):
        super().__init__()
        if not 0.0 < initial_depth_gate < 1.0:
            raise ValueError('initial_depth_gate must be strictly between 0 and 1')
        self.feature_dim = int(feature_dim)
        self.max_tokens = int(max_tokens)
        self.dino_norm = nn.LayerNorm(self.feature_dim)
        self.depth_norm = nn.LayerNorm(self.feature_dim)
        self.dino_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.depth_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.depth_gate_logit = nn.Parameter(
            torch.tensor(
                math.log(initial_depth_gate / (1.0 - initial_depth_gate)),
                dtype=torch.float32,
            )
        )
        self.out_norm = nn.LayerNorm(self.feature_dim)

    def _validate(
        self,
        dino_plan: torch.Tensor,
        depth_plan: torch.Tensor,
    ) -> None:
        if dino_plan.shape != depth_plan.shape:
            raise ValueError(
                'DINO and depth plans must have the same shape, got '
                f'{tuple(dino_plan.shape)} and {tuple(depth_plan.shape)}'
            )
        expected = (self.max_tokens, self.feature_dim)
        if dino_plan.ndim != 3 or tuple(dino_plan.shape[1:]) != expected:
            raise ValueError(
                f'each plan must have shape [B, {self.max_tokens}, '
                f'{self.feature_dim}], got {tuple(dino_plan.shape)}'
            )
        if not torch.isfinite(dino_plan).all():
            raise ValueError('DINO plan contains non-finite values')
        if not torch.isfinite(depth_plan).all():
            raise ValueError('depth plan contains non-finite values')

    def forward(
        self,
        dino_plan: torch.Tensor,
        depth_plan: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(dino_plan, depth_plan)
        dino = self.dino_proj(self.dino_norm(dino_plan))
        depth = self.depth_proj(self.depth_norm(depth_plan))
        gate = torch.sigmoid(self.depth_gate_logit).to(dtype=dino.dtype)
        return self.out_norm(dino + gate * depth)
```

- [ ] **Step 4: Attach fusion to the video expert**

Change the video expert constructor and factory:

```python
def __init__(self, net, semantic_plan_fusion=None):
    super().__init__()
    self.net = net
    self.semantic_plan_fusion = semantic_plan_fusion


def fuse_semantic_plan(self, dino_plan, depth_plan):
    if self.semantic_plan_fusion is None:
        raise RuntimeError(
            'online DINO+depth conditioning requires semantic_plan_fusion'
        )
    parameter = next(self.semantic_plan_fusion.parameters())
    return self.semantic_plan_fusion(
        dino_plan.to(device=parameter.device, dtype=parameter.dtype),
        depth_plan.to(device=parameter.device, dtype=parameter.dtype),
    )
```

Extend `from_pretrained` with:

```python
semantic_plan_fusion_enabled: bool = False,
semantic_plan_feature_dim: int = 1024,
semantic_plan_max_tokens: int = 1280,
semantic_plan_initial_depth_gate: float = 0.1,
```

When enabled, construct `DinoDepthPlanFusion`, move it with `.to(device=device, dtype=torch_dtype)`, and pass it to `cls`. Because it is assigned as a child of `CosmosVideoExpert`, it must appear under the video expert state dictionary.

- [ ] **Step 5: Test video-expert state ownership**

Append a test that stubs the heavy Cosmos imports, constructs the video expert with a fusion module, and checks:

```python
state_keys = set(video_expert.state_dict())
assert 'semantic_plan_fusion.depth_gate_logit' in state_keys
assert 'semantic_plan_fusion.dino_proj.weight' in state_keys
assert 'semantic_plan_fusion.depth_proj.weight' in state_keys
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest -q tests/test_fastwam_dino_depth_fusion.py
```

Expected: PASS.

- [ ] **Step 7: Commit fusion changes**

Run:

```bash
git add \
  third_party/FastWAM/src/fastwam/models/cosmos/semantic_plan_fusion.py \
  third_party/FastWAM/src/fastwam/models/cosmos/video_expert.py \
  tests/test_fastwam_dino_depth_fusion.py
git commit -m 'feat: fuse same-position dino and depth plans'
```

---

## Task 5: Make Effective Video FPS and Raw Instruction Dataset Outputs

**Files:**

- Modify: `third_party/FastWAM/src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
- Modify: `third_party/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py`
- Modify: `tests/test_fastwam_cosmos_semantic_plan.py`

- [ ] **Step 1: Add failing tests for the sampling-rate equation**

Add:

```python
def test_effective_video_fps_uses_both_sampling_strides():
    dataset_module = load_fastwam_module(
        'fastwam.datasets.lerobot.robot_video_dataset'
    )
    assert dataset_module.compute_effective_video_fps(
        raw_fps=20.0,
        global_sample_stride=2,
        action_video_freq_ratio=4,
    ) == pytest.approx(2.5)


def test_robot_video_sample_emits_fps_and_raw_instruction(fake_dataset):
    sample = fake_dataset[0]
    assert sample['video_fps'].ndim == 0
    assert sample['video_fps'].item() == pytest.approx(
        fake_dataset.lerobot_dataset.fps
        / (
            fake_dataset.global_sample_stride
            * fake_dataset.action_video_freq_ratio
        )
    )
    assert sample['instruction'] == fake_dataset.lerobot_dataset.task_text
```

Update the existing fake base dataset fixture to expose deterministic `fps` and `task_text` attributes.

- [ ] **Step 2: Run the dataset tests and confirm fields are missing**

Run:

```bash
pytest -q tests/test_fastwam_cosmos_semantic_plan.py \
  -k 'effective_video_fps or emits_fps'
```

Expected: FAIL because the helper and sample fields do not exist.

- [ ] **Step 3: Preserve raw FPS and compute effective FPS centrally**

Add at module scope in `robot_video_dataset.py`:

```python
def compute_effective_video_fps(
    *,
    raw_fps: float,
    global_sample_stride: int,
    action_video_freq_ratio: int,
) -> float:
    if raw_fps <= 0:
        raise ValueError(f'raw_fps must be positive, got {raw_fps}')
    if global_sample_stride <= 0:
        raise ValueError(
            f'global_sample_stride must be positive, got {global_sample_stride}'
        )
    if action_video_freq_ratio <= 0:
        raise ValueError(
            'action_video_freq_ratio must be positive, got '
            f'{action_video_freq_ratio}'
        )
    return float(raw_fps) / float(
        global_sample_stride * action_video_freq_ratio
    )
```

In `BaseLerobotDataset.__init__`, after validating consistent dataset FPS:

```python
self.fps = float(fps)
self.global_sample_stride = int(global_sample_stride)
```

In `RobotVideoDataset.__init__`:

```python
self.global_sample_stride = int(global_sample_stride)
self.video_fps = compute_effective_video_fps(
    raw_fps=self.lerobot_dataset.fps,
    global_sample_stride=self.global_sample_stride,
    action_video_freq_ratio=self.action_video_freq_ratio,
)
```

At sample construction:

```python
data['instruction'] = str(task)
data['video_fps'] = torch.tensor(self.video_fps, dtype=torch.float32)
```

Keep the existing model-facing formatted `prompt` unchanged.

- [ ] **Step 4: Add rejection tests for invalid rates**

Add:

```python
@pytest.mark.parametrize(
    ('raw_fps', 'global_stride', 'ratio'),
    [(0.0, 1, 4), (20.0, 0, 4), (20.0, 1, 0)],
)
def test_effective_video_fps_rejects_non_positive_inputs(
    raw_fps,
    global_stride,
    ratio,
):
    dataset_module = load_fastwam_module(
        'fastwam.datasets.lerobot.robot_video_dataset'
    )
    with pytest.raises(ValueError, match='positive'):
        dataset_module.compute_effective_video_fps(
            raw_fps=raw_fps,
            global_sample_stride=global_stride,
            action_video_freq_ratio=ratio,
        )
```

- [ ] **Step 5: Run the focused dataset tests**

Run:

```bash
pytest -q tests/test_fastwam_cosmos_semantic_plan.py \
  -k 'video_fps or raw_instruction'
```

Expected: PASS.

- [ ] **Step 6: Commit the data contract**

Run:

```bash
git add \
  third_party/FastWAM/src/fastwam/datasets/lerobot/base_lerobot_dataset.py \
  third_party/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py \
  tests/test_fastwam_cosmos_semantic_plan.py
git commit -m 'feat: expose sampled video timing to fastwam'
```

---

## Task 6: Invoke the Frozen Provider During FastWAM Training and Inference

**Files:**

- Create: `third_party/FastWAM/src/fastwam/models/cosmos/online_semantic_planner.py`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/runtime.py`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py`
- Modify: `tests/test_fastwam_cosmos_semantic_plan.py`

- [ ] **Step 1: Add failing online-training routing tests**

Use a plain fake provider and a fake video expert with a trainable fusion scalar:

```python
class FakeOnlinePlan:
    def __init__(self, batch):
        self.dino_plan = torch.ones(batch, 1280, 1024)
        self.depth_plan = torch.full((batch, 1280, 1024), 2.0)
        self.semantic_plan_times = torch.tensor(
            [[0.125, 0.375, 0.5, 0.75, 1.0]]
        ).expand(batch, -1)


class FakeOnlineProvider:
    def __init__(self):
        self.calls = []

    def predict(self, images, instructions):
        self.calls.append((images.detach().clone(), list(instructions)))
        return FakeOnlinePlan(images.shape[0])


def test_training_uses_current_rgb_and_raw_instruction_online(
    fastwam_model,
    training_sample,
):
    provider = FakeOnlineProvider()
    fastwam_model._online_semantic_planner = provider
    training_sample['instruction'] = ['open drawer']
    training_sample['video_fps'] = torch.tensor([5.0])

    fastwam_model.training_loss(training_sample)

    assert len(provider.calls) == 1
    images, instructions = provider.calls[0]
    assert torch.equal(images, training_sample['video'][:, :, 0])
    assert instructions == ['open drawer']
    assert fastwam_model._current_semantic_plan.shape == (1, 1280, 1024)
    assert fastwam_model._current_semantic_plan_times.shape == (1, 5)
    assert fastwam_model._current_video_fps.item() == pytest.approx(5.0)
```

The fake fusion method should return a tensor containing a trainable parameter, and the test should additionally assert that backward populates its gradient while all fake provider tensors remain detached.

- [ ] **Step 2: Add failing exclusivity and required-field tests**

Add:

```python
def test_online_and_file_backed_semantics_are_mutually_exclusive(
    fastwam_model,
    training_sample,
):
    fastwam_model._online_semantic_planner = FakeOnlineProvider()
    training_sample['semantic_plan'] = torch.zeros(1, 1280, 1024)
    training_sample['semantic_plan_times'] = torch.zeros(1, 5)
    with pytest.raises(ValueError, match='mutually exclusive'):
        fastwam_model.training_loss(training_sample)


@pytest.mark.parametrize('missing_field', ['instruction', 'video_fps'])
def test_online_training_requires_instruction_and_fps(
    fastwam_model,
    training_sample,
    missing_field,
):
    fastwam_model._online_semantic_planner = FakeOnlineProvider()
    training_sample['instruction'] = ['open drawer']
    training_sample['video_fps'] = torch.tensor([5.0])
    del training_sample[missing_field]
    with pytest.raises(KeyError, match=missing_field):
        fastwam_model.training_loss(training_sample)


def test_online_planner_is_not_registered_in_fastwam_state_dict(
    fastwam_model_factory,
):
    class ModuleProvider(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

    provider = ModuleProvider()
    model = fastwam_model_factory(online_semantic_planner=provider)
    assert model._online_semantic_planner is provider
    assert not any(
        '_online_semantic_planner' in key
        for key in model.state_dict()
    )
```

- [ ] **Step 3: Run the routing tests and confirm they fail**

Run:

```bash
pytest -q tests/test_fastwam_cosmos_semantic_plan.py \
  -k 'online or mutually_exclusive'
```

Expected: FAIL because FastWAM only accepts file-backed tensors.

- [ ] **Step 4: Add an import-safe provider loader**

Create:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_online_semantic_planner(
    *,
    code_dir: str,
    checkpoint_dir: str,
    device,
    dtype,
):
    code_dir = Path(code_dir).resolve()
    module_path = code_dir / 'dino_depth_plan_provider.py'
    if not module_path.is_file():
        raise FileNotFoundError(
            f'online planner provider not found: {module_path}'
        )
    trainer_dir = code_dir.parent
    if str(trainer_dir) not in sys.path:
        sys.path.insert(0, str(trainer_dir))
    spec = importlib.util.spec_from_file_location(
        'fastwam_dino_depth_plan_provider',
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f'cannot load provider module from {module_path}')
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FrozenDinoDepthPlanProvider.from_checkpoint(
        checkpoint_dir,
        device=device,
        dtype=dtype,
    )
```

This loader is the only FastWAM file that knows the external planner code location.

- [ ] **Step 5: Build the provider in runtime without module registration**

Extend the flat, Hydra-instantiated `create_fastwam_cosmos` signature with:

```python
online_semantic_planner: bool = False,
online_semantic_planner_code_dir: str | None = None,
online_semantic_planner_checkpoint: str | None = None,
semantic_plan_initial_depth_gate: float = 0.1,
```

Then make the factory:

1. Enables the video-expert fusion module whenever `online_semantic_planner.enabled` is true.
2. Constructs the provider after resolving the local rank/device.
3. Passes the plain provider into `FastWAMCosmos`.

Use:

```python
online_enabled = bool(online_semantic_planner)
if online_enabled and not online_semantic_planner_code_dir:
    raise ValueError(
        'online_semantic_planner_code_dir is required when online mode is enabled'
    )
if online_enabled and not online_semantic_planner_checkpoint:
    raise ValueError(
        'online_semantic_planner_checkpoint is required when online mode is enabled'
    )

video_expert = CosmosVideoExpert.from_pretrained(
    ckpt_path=video_dit_pretrained_path,
    atten_backend=atten_backend,
    device=device,
    torch_dtype=model_dtype,
    semantic_plan_context=semantic_plan_context,
    semantic_plan_in_dim=semantic_plan_in_dim,
    semantic_plan_hidden_dim=semantic_plan_hidden_dim,
    semantic_plan_max_tokens=semantic_plan_max_tokens,
    semantic_plan_num_keyframes=semantic_plan_num_keyframes,
    semantic_plan_source_num_keyframes=semantic_plan_source_num_keyframes,
    semantic_plan_spatial_grid=semantic_plan_spatial_grid,
    semantic_plan_coord_hidden_dim=semantic_plan_coord_hidden_dim,
    semantic_plan_use_rope=semantic_plan_use_rope,
    semantic_plan_cross_attention_blocks=(
        semantic_plan_cross_attention_blocks
    ),
    semantic_plan_fusion_enabled=online_enabled,
    semantic_plan_feature_dim=int(semantic_plan_in_dim),
    semantic_plan_max_tokens=int(semantic_plan_max_tokens),
    semantic_plan_initial_depth_gate=float(semantic_plan_initial_depth_gate),
)

online_provider = None
if online_enabled:
    online_provider = load_online_semantic_planner(
        code_dir=str(online_semantic_planner_code_dir),
        checkpoint_dir=str(online_semantic_planner_checkpoint),
        device=device,
        dtype=model_dtype,
    )
```

Pass `online_semantic_planner=online_provider` into the `FastWAMCosmos` constructor.

Assign the provider inside `FastWAMCosmos.__init__` with:

```python
object.__setattr__(
    self,
    '_online_semantic_planner',
    online_semantic_planner,
)
```

This prevents a future provider implementation that subclasses `nn.Module` from being registered under FastWAM accidentally.

- [ ] **Step 6: Implement a single semantic-condition preparation path**

Add these transient attributes:

```python
self.semantic_plan_dim = int(semantic_plan_dim)
self.semantic_plan_max_tokens = int(semantic_plan_max_tokens)
self.semantic_plan_num_keyframes = int(semantic_plan_num_keyframes)
self._current_semantic_plan = None
self._current_semantic_plan_times = None
self._current_video_fps = None
```

Add matching constructor parameters to `FastWAMCosmos` and pass the three existing factory values from `create_fastwam_cosmos`.

Add this validator and use it for both online and file-backed tensors:

```python
def _validate_semantic_tensors(self, plan, times, batch_size):
    expected_plan = (
        batch_size,
        self.semantic_plan_max_tokens,
        self.semantic_plan_dim,
    )
    expected_times = (batch_size, self.semantic_plan_num_keyframes)
    if tuple(plan.shape) != expected_plan:
        raise ValueError(
            f'semantic_plan must have shape {expected_plan}, '
            f'got {tuple(plan.shape)}'
        )
    if tuple(times.shape) != expected_times:
        raise ValueError(
            f'semantic_plan_times must have shape {expected_times}, '
            f'got {tuple(times.shape)}'
        )
    if not torch.isfinite(plan).all():
        raise ValueError('semantic_plan contains non-finite values')
    if not torch.isfinite(times).all():
        raise ValueError('semantic_plan_times contains non-finite values')
    if (times < 0).any() or (times > 1).any():
        raise ValueError('semantic_plan_times must lie in [0, 1]')
    if not torch.all(times[:, 1:] > times[:, :-1]):
        raise ValueError('semantic_plan_times must be strictly increasing')
```

Add:

```python
def _prepare_semantic_condition(self, sample, current_rgb):
    has_offline_plan = sample.get('semantic_plan') is not None
    has_online_provider = self._online_semantic_planner is not None
    if has_offline_plan and has_online_provider:
        raise ValueError(
            'online and file-backed semantic plans are mutually exclusive'
        )

    video_fps = sample.get('video_fps')
    if video_fps is None:
        if has_online_provider or has_offline_plan:
            raise KeyError('video_fps is required for semantic conditioning')
        self._current_video_fps = None
    else:
        video_fps = torch.as_tensor(
            video_fps,
            device=current_rgb.device,
            dtype=torch.float32,
        )
        if video_fps.ndim == 0:
            video_fps = video_fps.expand(current_rgb.shape[0])
        if video_fps.shape != (current_rgb.shape[0],):
            raise ValueError(
                f'video_fps must have shape [{current_rgb.shape[0]}], '
                f'got {tuple(video_fps.shape)}'
            )
        if not torch.isfinite(video_fps).all() or (video_fps <= 0).any():
            raise ValueError('video_fps must contain finite positive values')
        self._current_video_fps = video_fps

    if has_online_provider:
        instructions = sample.get('instruction')
        if instructions is None:
            raise KeyError(
                'instruction is required for online semantic conditioning'
            )
        if isinstance(instructions, str):
            instructions = [instructions]
        result = self._online_semantic_planner.predict(
            current_rgb,
            list(instructions),
        )
        self._current_semantic_plan = self.video_expert.fuse_semantic_plan(
            result.dino_plan,
            result.depth_plan,
        )
        self._current_semantic_plan_times = (
            result.semantic_plan_times.to(
                device=self._current_semantic_plan.device,
                dtype=torch.float32,
            )
        )
        self._validate_semantic_tensors(
            self._current_semantic_plan,
            self._current_semantic_plan_times,
            current_rgb.shape[0],
        )
        expected_times = self._current_semantic_plan_times.new_tensor(
            [0.125, 0.375, 0.5, 0.75, 1.0]
        ).unsqueeze(0).expand(current_rgb.shape[0], -1)
        if not torch.allclose(
            self._current_semantic_plan_times,
            expected_times,
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError(
                'online semantic_plan_times must equal '
                '[0.125, 0.375, 0.5, 0.75, 1.0]'
            )
        return

    self._set_current_semantic_plan(sample)
```

Validate offline shapes against the configured geometry in `_set_current_semantic_plan` rather than silently accepting arbitrary token counts.

Extend `build_inputs` so training can reuse an already moved video without a second device copy:

```python
def build_inputs(self, sample, tiled: bool = False, video=None):
    if video is None:
        video = sample['video'].to(
            device=self.device,
            dtype=self.torch_dtype,
            non_blocking=True,
        )
```

This replaces only the current unconditional first `video = sample['video'].to(...)` assignment; the existing context, action, padding, proprio, VAE encode, and return dictionary stay byte-for-byte unchanged.

At the start of `training_loss`, invoke the planner before VAE encoding:

```python
raw_video = sample['video'].to(
    device=self.device,
    dtype=self.torch_dtype,
    non_blocking=True,
)
self._prepare_semantic_condition(
    sample,
    raw_video[:, :, 0],
)
inp = self.build_inputs(sample, tiled=tiled, video=raw_video)
```

Delete the old `inp = self.build_inputs(...)` and `_set_current_semantic_plan(sample)` calls so each operation occurs once. The provider consumes the same horizontally composed current frame used by FastWAM, and the Wan VAE still receives all nine RGB frames.

- [ ] **Step 7: Add online inference to `infer_action`**

Extend the signature with:

```python
instruction: str | list[str] | None = None,
video_fps: float | torch.Tensor | None = None,
```

Before VAE encoding, create the same sample-shaped conditioning dictionary and call `_prepare_semantic_condition`. When the online provider is enabled:

- `instruction` is required even if precomputed text `context` is supplied.
- `video_fps` is required.
- Direct `semantic_plan` arguments are rejected as mutually exclusive.

Use the already normalized/composed `input_image` tensor as `current_rgb`; do not encode/decode an image solely for the provider.

`instruction` is always the raw robot task for the 4B planner. `prompt` retains the existing FastWAM formatted text-cache string; do not substitute one for the other internally.

- [ ] **Step 8: Run training and inference routing tests**

Run:

```bash
pytest -q tests/test_fastwam_cosmos_semantic_plan.py \
  -k 'online or mutually_exclusive or requires_instruction'
```

Expected: PASS.

- [ ] **Step 9: Commit the online model integration**

Run:

```bash
git add \
  third_party/FastWAM/src/fastwam/models/cosmos/online_semantic_planner.py \
  third_party/FastWAM/src/fastwam/models/cosmos/runtime.py \
  third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py \
  tests/test_fastwam_cosmos_semantic_plan.py
git commit -m 'feat: run frozen semantic planner inside fastwam'
```

---

## Task 7: Route Sampled FPS Through Every Cosmos Semantic-Attention Path

**Files:**

- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/couplings/mot.py`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/couplings/cross_attn.py`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/couplings/agra.py`
- Modify: `third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py`
- Modify: `third_party/FastWAM/configs/model/fastwam_cosmos.yaml`
- Modify: `third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml`
- Create: `third_party/FastWAM/configs/task/libero_cosmos_2cam224_online_dino_depth.yaml`
- Modify: `third_party/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py`
- Modify: `tests/test_fastwam_cosmos_semantic_plan.py`

- [ ] **Step 1: Add failing tests that capture FPS at all video-expert calls**

Instrument the fake video expert to record the `fps` keyword, then add parametrized coverage:

```python
@pytest.mark.parametrize(
    ('coupling_name', 'expected_calls'),
    [
        ('mot', 1),
        ('cross_attn', 1),
        ('agra', 2),
    ],
)
def test_couplings_forward_sampled_fps_to_video_expert(
    coupling_name,
    expected_calls,
    coupling_fixture,
):
    coupling, model, inputs = coupling_fixture(coupling_name)
    model._current_video_fps = torch.tensor([5.0, 5.0])
    coupling.loss(model, **inputs)
    assert model.video_expert.recorded_fps == [
        model._current_video_fps
    ] * expected_calls
```

For AGRA, the two expected calls are the standalone video loss and the foresight call. If the current fixture exercises multiple foresight evaluations, assert that exact count and verify every element is the identical FPS tensor.

- [ ] **Step 2: Run coupling tests and confirm FPS is dropped**

Run:

```bash
pytest -q tests/test_fastwam_cosmos_semantic_plan.py \
  -k couplings_forward_sampled_fps
```

Expected: FAIL because the coupling calls omit `fps`.

- [ ] **Step 3: Pass FPS explicitly at each call site**

In `mot.py`:

```python
v = model.video_expert.prepare(
    noisy_latents,
    t_v,
    crossattn_emb,
    o0_latent=o0,
    cond_frames=cond_frames,
    fps=model._current_video_fps,
    semantic_plan_B_L_D=model._current_semantic_plan,
    semantic_plan_times_B_N=model._current_semantic_plan_times,
)
```

In `cross_attn.py`:

```python
pred_v, vfeat = model.video_expert.forward_standalone(
    noisy_latents,
    t_v,
    crossattn_emb,
    feature_layer=model.feature_layer,
    fps=model._current_video_fps,
    semantic_plan_B_L_D=model._current_semantic_plan,
    semantic_plan_times_B_N=model._current_semantic_plan_times,
)
```

In both `agra.py` call sites:

```python
fps=model._current_video_fps,
```

In the AGRA-specific inference `forward_foresight` call inside `fastwam_cosmos.py`, also pass:

```python
fps=self._current_video_fps,
```

Do not fall back to a hard-coded `16` in a coupling. The video expert may retain its legacy fallback only for non-semantic callers; semantic conditioning itself requires FPS in `FastWAMCosmos`.

- [ ] **Step 4: Replace stale semantic geometry and add online config**

Replace the stale flat semantic fields in `fastwam_cosmos.yaml` with:

```yaml
semantic_plan_context: true
semantic_plan_in_dim: 1024
semantic_plan_hidden_dim: 2048
semantic_plan_num_keyframes: 5
semantic_plan_source_num_keyframes: 5
semantic_plan_spatial_grid: 16
semantic_plan_max_tokens: 1280
semantic_plan_coord_hidden_dim: 256
semantic_plan_use_rope: true
semantic_plan_cross_attention_blocks: null

online_semantic_planner: false
online_semantic_planner_code_dir: scripts/qwen3_vl_semantic_planner/lingbot_dino_4b
online_semantic_planner_checkpoint: null
semantic_plan_initial_depth_gate: 0.1
```

`semantic_plan_context` enables the adapter geometry, while `online_semantic_planner` chooses the source. A file-backed dataset can therefore use the same adapter with online mode off.

Add a flat constructor argument `semantic_plan_source: Literal['none', 'online', 'file'] = 'none'` to `RobotVideoDataset`. Validate:

- `none` requires both file-backed paths to be null.
- `online` requires both file-backed paths to be null and emits no offline tensor.
- `file` requires a manifest and keeps the existing loading behavior.
- Every other value raises `ValueError`.

In the base `libero_2cam_cosmos.yaml`, make the neutral source and new geometry explicit:

```yaml
semantic_plan_source: none
semantic_plan_dir: null
semantic_plan_manifest: null
semantic_plan_dim: 1024
semantic_plan_max_tokens: 1280
semantic_plan_default_to_zero: false
```

Create the dedicated online task config:

```yaml
# @package _global_

defaults:
  - libero_cosmos_2cam224
  - _self_

model:
  online_semantic_planner: true
  online_semantic_planner_checkpoint: ${oc.env:FASTWAM_PLANNER_CHECKPOINT}

data:
  train:
    semantic_plan_source: online
```

The existing task remains available for no-plan or explicitly file-backed runs.

- [ ] **Step 5: Add exact YAML regression tests**

Add:

```python
def test_cosmos_config_uses_fastwam_planner_geometry():
    config = yaml.safe_load(
        (
            ROOT
            / 'third_party/FastWAM/configs/model/fastwam_cosmos.yaml'
        ).read_text()
    )
    assert config['semantic_plan_context'] is True
    assert config['semantic_plan_in_dim'] == 1024
    assert config['semantic_plan_hidden_dim'] == 2048
    assert config['semantic_plan_num_keyframes'] == 5
    assert config['semantic_plan_source_num_keyframes'] == 5
    assert config['semantic_plan_spatial_grid'] == 16
    assert config['semantic_plan_max_tokens'] == 1280
    assert config['semantic_plan_coord_hidden_dim'] == 256
    assert config['semantic_plan_use_rope'] is True
    assert config['semantic_plan_cross_attention_blocks'] is None
    assert config['online_semantic_planner'] is False
    assert config['semantic_plan_initial_depth_gate'] == pytest.approx(0.1)
```

- [ ] **Step 6: Run coupling and configuration tests**

Run:

```bash
pytest -q tests/test_fastwam_cosmos_semantic_plan.py \
  -k 'couplings_forward_sampled_fps or planner_geometry or source_mode'
```

Expected: PASS.

- [ ] **Step 7: Commit timing and configuration changes**

Run:

```bash
git add \
  third_party/FastWAM/src/fastwam/models/cosmos/couplings/mot.py \
  third_party/FastWAM/src/fastwam/models/cosmos/couplings/cross_attn.py \
  third_party/FastWAM/src/fastwam/models/cosmos/couplings/agra.py \
  third_party/FastWAM/src/fastwam/models/cosmos/fastwam_cosmos.py \
  third_party/FastWAM/configs/model/fastwam_cosmos.yaml \
  third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
  third_party/FastWAM/configs/task/libero_cosmos_2cam224_online_dino_depth.yaml \
  third_party/FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py \
  tests/test_fastwam_cosmos_semantic_plan.py
git commit -m 'feat: route semantic timing through cosmos'
```

---

## Task 8: Add Checkpoint-Backed Smoke Verification and Run the Full Gate

**Files:**

- Create: `third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py`
- Modify: `tests/test_fastwam_cosmos_semantic_plan.py`
- Modify: `tests/test_dino_depth_plan_provider.py` if smoke-discovered contract coverage is missing

- [ ] **Step 1: Add a smoke script argument-contract test**

Add:

```python
def test_online_smoke_script_requires_real_checkpoint_and_config():
    smoke_path = (
        ROOT
        / 'third_party/FastWAM/scripts'
        / 'smoke_online_dino_depth_semantic_plan.py'
    )
    source = smoke_path.read_text()
    assert '--planner-checkpoint' in source
    assert '--config-dir' in source
    assert '--config-name' in source
    assert '--task' in source
    assert 'torch.inference_mode()' in source
```

- [ ] **Step 2: Create the one-batch smoke script**

The script must:

1. Parse `--planner-checkpoint`, `--config-dir`, `--config-name`, `--task`, `--device`, `--instruction`, `--image`, and `--video-fps`. Default `--task` to `libero_cosmos_2cam224_online_dino_depth`.
2. Verify the planner checkpoint with `validate_checkpoint_files` and `validate_planner_metadata` before loading FastWAM.
3. Compose the Hydra config with `overrides=[f'task={args.task}']` and force:

```python
cfg.model.online_semantic_planner = True
cfg.model.online_semantic_planner_checkpoint = str(
    args.planner_checkpoint
)
```

4. Load one RGB image, normalize it to `[-1, 1]`, and call `infer_action` with the raw instruction and explicit sampled-video FPS.
5. Register a temporary forward hook on `video_expert.semantic_plan_fusion` and assert its output shape is exactly `(1, 1280, 1024)`.
6. Assert the action output is finite and has a non-empty temporal dimension.
7. Print only the verified checkpoint path, fused-plan shape, action shape, and sampled FPS.

Use this main structure:

```python
def main():
    args = parse_args()
    validate_checkpoint(args.planner_checkpoint)
    cfg = load_config(args)
    model = create_fastwam_cosmos(cfg).eval()
    image = load_rgb_tensor(args.image, device=args.device)
    captured = {}

    def capture_fused_plan(_module, _inputs, output):
        captured['shape'] = tuple(output.shape)

    handle = model.video_expert.semantic_plan_fusion.register_forward_hook(
        capture_fused_plan
    )
    try:
        with torch.inference_mode():
            result = model.infer_action(
                input_image=image,
                instruction=args.instruction,
                prompt=(
                    "A video recorded from a robot's point of view executing "
                    f'the following instruction: {args.instruction}'
                ),
                video_fps=args.video_fps,
            )
            actions = result['action']
    finally:
        handle.remove()

    if captured.get('shape') != (1, 1280, 1024):
        raise RuntimeError(
            f'unexpected fused plan shape: {captured.get("shape")}'
        )
    if actions.numel() == 0 or not torch.isfinite(actions).all():
        raise RuntimeError('action output is empty or non-finite')
    print(
        {
            'planner_checkpoint': str(args.planner_checkpoint),
            'fused_plan_shape': captured['shape'],
            'action_shape': tuple(actions.shape),
            'video_fps': float(args.video_fps),
        }
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run all CPU-safe unit tests**

Run:

```bash
pytest -q \
  tests/test_lingbot_dino_depth_contract.py \
  tests/test_dino_depth_plan_provider.py \
  tests/test_fastwam_dino_depth_fusion.py \
  tests/test_fastwam_cosmos_semantic_plan.py \
  tests/test_cosmos_semantic_plan_stage2.py
```

Expected: PASS with no skipped test that covers a contract listed in this plan.

- [ ] **Step 4: Run syntax and import checks**

Run:

```bash
python -m compileall -q \
  scripts/qwen3_vl_semantic_planner \
  third_party/FastWAM/src/fastwam/models/cosmos \
  third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py
bash -n \
  scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k5.sh
```

Expected: both commands exit with code 0.

- [ ] **Step 5: Train or fine-tune the nine-frame planner**

Run with the actual dataset and output locations:

```bash
FASTWAM_DATA_CONFIG=third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml \
FASTWAM_DATASET_DIRS=/data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_spatial_no_noops_lerobot:/data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_object_no_noops_lerobot:/data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_goal_no_noops_lerobot:/data/LFT-W02_data/junjie/data/LIBERO-fastwam/libero_10_no_noops_lerobot \
OUTPUT_DIR=/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/qwen3vl4b_dino_depth_fastwam_k5 \
scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k5.sh
```

Expected export under the step directory named by `OUTPUT_DIR/latest_checkpoint.txt`:

```text
qwen3vl_lora_or_model/
processor/
plan_head.pt
depth_head.pt
plan_token_embedding.pt
planner_meta.json
```

Resolve the per-step directory and validate its `planner_meta.json`:

```bash
PLANNER_CHECKPOINT=$(tr -d '\n' < \
  /data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/qwen3vl4b_dino_depth_fastwam_k5/latest_checkpoint.txt)
test -d "$PLANNER_CHECKPOINT"
```

The metadata must report:

```json
{
  "sequence_length": 9,
  "num_keyframes": 5,
  "grid_size": 16,
  "semantic_dim": 1024,
  "target_tokens": 1280,
  "keyframe_offsets": [1, 3, 4, 6, 8],
  "has_depth_head": true,
  "token_order": "keyframe_major_row_major"
}
```

- [ ] **Step 6: Run the checkpoint-backed FastWAM smoke test**

Use an image whose horizontal composition matches the two-camera FastWAM input:

```bash
PLANNER_CHECKPOINT=$(tr -d '\n' < \
  /data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/outputs/qwen3vl4b_dino_depth_fastwam_k5/latest_checkpoint.txt)
FASTWAM_PLANNER_CHECKPOINT="$PLANNER_CHECKPOINT" \
python third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py \
  --planner-checkpoint \
  "$PLANNER_CHECKPOINT" \
  --config-dir third_party/FastWAM/configs \
  --config-name train \
  --task libero_cosmos_2cam224_online_dino_depth \
  --device cuda:0 \
  --instruction 'open the middle drawer' \
  --image /absolute/path/to/two_camera_current_frame.png \
  --video-fps 5.0
```

Expected: one printed dictionary with `fused_plan_shape=(1, 1280, 1024)` and a finite, non-empty action shape.

- [ ] **Step 7: Inspect the final diff for scope and accidental placeholders**

Run:

```bash
git status --short
git diff --check
rg -n 'TODO|TBD|NotImplemented|pass$' \
  scripts/qwen3_vl_semantic_planner \
  third_party/FastWAM/src/fastwam/models/cosmos \
  third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py \
  tests/test_lingbot_dino_depth_contract.py \
  tests/test_dino_depth_plan_provider.py \
  tests/test_fastwam_dino_depth_fusion.py \
  tests/test_fastwam_cosmos_semantic_plan.py
```

Expected:

- `git diff --check` produces no output.
- The placeholder scan produces no new placeholder in modified lines.
- `git status --short` contains only intentional work plus pre-existing user changes.

- [ ] **Step 8: Commit the smoke test**

Run:

```bash
git add \
  third_party/FastWAM/scripts/smoke_online_dino_depth_semantic_plan.py \
  tests/test_fastwam_cosmos_semantic_plan.py \
  tests/test_dino_depth_plan_provider.py
git commit -m 'test: verify online dino depth fastwam path'
```

---

## Final Acceptance Checklist

- [ ] A production planner checkpoint was trained or fine-tuned with `sequence_length=9` and exports both `plan_head.pt` and `depth_head.pt`.
- [ ] The provider performs one Qwen forward for both branches and returns detached `[B,1280,1024]` DINO and depth tensors.
- [ ] FastWAM invokes the provider online in training and inference from the current composed RGB image and raw instruction.
- [ ] The planner remains frozen and is not registered in `FastWAMCosmos.state_dict()`.
- [ ] The fusion module is registered under the Cosmos video expert, receives gradients, and starts with a depth contribution gate of approximately `0.1`.
- [ ] Semantic keyframe times are exactly `[0.125,0.375,0.5,0.75,1.0]` for offsets `[1,3,4,6,8]`.
- [ ] Effective sampled-video FPS is emitted by the dataset and reaches MoT, cross-attention, AGRA standalone video loss, AGRA foresight, and inference.
- [ ] Online and file-backed sources cannot be active together.
- [ ] The full CPU-safe test command, syntax checks, and checkpoint-backed GPU smoke test pass.
