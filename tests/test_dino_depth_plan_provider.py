from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROVIDER_PATH = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
    / "dino_depth_plan_provider.py"
)
TRAINER_PATH = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner"
    / "train_qwen3vl4b_lingbot_dino_planner.py"
)

RECONSTRUCTION_METADATA_FIELDS = (
    "semantic_dim",
    "plan_token_ids",
    "target_tokens",
    "num_keyframes",
    "grid_size",
    "branch_latent_per_keyframe",
    "shared_latent_per_keyframe",
    "private_latent_per_keyframe",
    "plan_head_type",
    "plan_head_num_heads",
    "plan_head_dropout",
    "sem_mlp_hidden_size",
    "mse_loss_weight",
    "cosine_loss_weight",
    "norm_loss_weight",
    "variance_loss_weight",
    "infonce_loss_weight",
    "infonce_temperature",
    "depth_feature_dim",
    "depth_grid_size",
    "depth_loss_weight",
    "use_current_alignment",
    "num_task_tokens",
    "current_dino_loss_weight",
    "future_dino_loss_weight",
    "current_depth_loss_weight",
    "future_depth_loss_weight",
)

NUMERIC_METADATA_FIELDS = (
    "sequence_length",
    "num_keyframes",
    "grid_size",
    "semantic_dim",
    "target_tokens",
    "depth_feature_dim",
    "depth_grid_size",
    "shared_latent_per_keyframe",
    "private_latent_per_keyframe",
    "branch_latent_per_keyframe",
    "total_unique_latent_per_keyframe",
    "latent_len",
    "plan_head_num_heads",
    "plan_head_dropout",
    "sem_mlp_hidden_size",
    "mse_loss_weight",
    "cosine_loss_weight",
    "norm_loss_weight",
    "variance_loss_weight",
    "infonce_loss_weight",
    "infonce_temperature",
    "depth_loss_weight",
    "num_task_tokens",
    "current_dino_loss_weight",
    "future_dino_loss_weight",
    "current_depth_loss_weight",
    "future_depth_loss_weight",
)


def load_provider_module():
    spec = importlib.util.spec_from_file_location("dino_depth_provider", PROVIDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_trainer_module():
    trainer_dir = str(TRAINER_PATH.parent)
    if trainer_dir not in sys.path:
        sys.path.insert(0, trainer_dir)
    spec = importlib.util.spec_from_file_location(
        "dino_depth_provider_trainer", TRAINER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_metadata():
    return {
        "sequence_length": 9,
        "num_keyframes": 1,
        "grid_size": 16,
        "semantic_dim": 1024,
        "target_tokens": 256,
        "keyframe_offsets": [8],
        "keyframe_scheme": "even_future",
        "normalized_keyframe_times": [1.0],
        "has_depth_head": True,
        "depth_feature_dim": 1024,
        "depth_grid_size": 16,
        "shared_latent_per_keyframe": 32,
        "private_latent_per_keyframe": 32,
        "branch_latent_per_keyframe": 8,
        "total_unique_latent_per_keyframe": 16,
        "latent_len": 16,
        "use_current_alignment": True,
        "num_task_tokens": 8,
        "query_layout": "current_8_then_future_8__dino_depth_shared_within_time",
        "plan_head_type": "lingbot_dino",
        "plan_head_num_heads": 16,
        "plan_head_dropout": 0.0,
        "sem_mlp_hidden_size": 0,
        "mse_loss_weight": 1.0,
        "cosine_loss_weight": 0.0,
        "norm_loss_weight": 0.0,
        "variance_loss_weight": 0.0,
        "infonce_loss_weight": 0.0,
        "infonce_temperature": 0.07,
        "depth_loss_weight": 0.004,
        "current_dino_loss_weight": 0.004,
        "future_dino_loss_weight": 0.004,
        "current_depth_loss_weight": 0.004,
        "future_depth_loss_weight": 0.004,
        "plan_token_ids": list(range(3, 19)),
        "planner_input_frame": "fastwam_current_multicamera_composite",
        "plan_token_strings": [f"<|sem_plan_{index}|>" for index in range(16)],
        "token_order": "keyframe_major_row_major",
    }


def independent_query_metadata():
    metadata = valid_metadata()
    metadata.update(
        {
            "independent_modality_task_tokens": True,
            "total_unique_latent_per_keyframe": 32,
            "latent_len": 32,
            "query_layout": (
                "current_dino_8_then_future_dino_8_then_"
                "current_depth_8_then_future_depth_8"
            ),
            "plan_token_ids": list(range(3, 35)),
            "plan_token_strings": [
                f"<|sem_plan_{index}|>" for index in range(32)
            ],
        }
    )
    return metadata


def independent_64_query_metadata():
    metadata = valid_metadata()
    metadata.update(
        {
            "independent_modality_task_tokens": True,
            "num_task_tokens": 64,
            "branch_latent_per_keyframe": 64,
            "total_unique_latent_per_keyframe": 256,
            "latent_len": 256,
            "query_layout": (
                "current_dino_64_then_future_dino_64_then_"
                "current_depth_64_then_future_depth_64"
            ),
            "plan_token_ids": list(range(3, 259)),
            "plan_token_strings": [
                f"<|sem_plan_{index}|>" for index in range(256)
            ],
        }
    )
    return metadata


def test_validate_metadata_accepts_exact_fastwam_contract():
    module = load_provider_module()
    contract = module.validate_planner_metadata(valid_metadata())
    assert contract.keyframe_offsets == (8,)
    assert contract.normalized_keyframe_times == (1.0,)


def test_validate_metadata_accepts_independent_modality_query_contract():
    module = load_provider_module()

    contract = module.validate_planner_metadata(independent_query_metadata())

    assert contract.total_unique_latent_per_keyframe == 32
    assert len(contract.plan_token_strings) == 32


def test_validate_metadata_accepts_64_tokens_per_independent_feature():
    module = load_provider_module()

    contract = module.validate_planner_metadata(independent_64_query_metadata())

    assert contract.num_task_tokens == 64
    assert contract.branch_latent_per_keyframe == 64
    assert contract.total_unique_latent_per_keyframe == 256
    assert len(contract.plan_token_strings) == 256


def test_validate_metadata_accepts_finite_informational_loss_values():
    module = load_provider_module()
    metadata = valid_metadata()
    metadata.update(
        {
            "mse_loss_weight": 0.75,
            "cosine_loss_weight": 0.25,
            "norm_loss_weight": 0.125,
            "variance_loss_weight": 0.0625,
            "infonce_loss_weight": 0.03125,
            "infonce_temperature": 0.2,
            "depth_loss_weight": 0.01,
        }
    )

    module.validate_planner_metadata(metadata)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sequence_length", 49),
        ("num_keyframes", 5),
        ("grid_size", 9),
        ("semantic_dim", 1152),
        ("target_tokens", 486),
        ("keyframe_offsets", [1, 3, 6, 8]),
        ("keyframe_scheme", "uniform"),
        ("has_depth_head", False),
        ("shared_latent_per_keyframe", 8),
        ("private_latent_per_keyframe", 64),
        ("query_layout", "keyframe_major"),
        ("latent_len", 256),
        ("plan_head_num_heads", 8),
        ("plan_head_dropout", 0.1),
        ("sem_mlp_hidden_size", 512),
        ("planner_input_frame", "legacy_single_current_frame"),
        ("token_order", "spatial_major"),
    ],
)
def test_validate_metadata_rejects_incompatible_checkpoint(field, value):
    module = load_provider_module()
    metadata = valid_metadata()
    metadata[field] = value
    with pytest.raises(ValueError, match=field):
        module.validate_planner_metadata(metadata)


@pytest.mark.parametrize(
    "field",
    ["normalized_keyframe_times", "plan_token_strings"],
)
def test_validate_metadata_names_malformed_sequence_field(field):
    module = load_provider_module()
    metadata = valid_metadata()
    metadata[field] = None

    with pytest.raises(ValueError, match=field):
        module.validate_planner_metadata(metadata)


def test_validate_metadata_rejects_non_finite_keyframe_time():
    module = load_provider_module()
    metadata = valid_metadata()
    metadata["normalized_keyframe_times"][0] = float("nan")

    with pytest.raises(ValueError, match="normalized_keyframe_times"):
        module.validate_planner_metadata(metadata)


@pytest.mark.parametrize(
    "invalid_time",
    [
        pytest.param(True, id="boolean"),
        pytest.param("1.0", id="wrong-type"),
    ],
)
def test_from_checkpoint_rejects_malformed_keyframe_time_before_loading(
    tmp_path,
    monkeypatch,
    invalid_time,
):
    module = load_provider_module()
    metadata = valid_metadata()
    metadata["normalized_keyframe_times"][-1] = invalid_time
    write_complete_checkpoint_layout(tmp_path, metadata)
    load_counts = install_forbidden_checkpoint_loaders(monkeypatch)

    with pytest.raises(ValueError, match="normalized_keyframe_times"):
        module.FrozenDinoDepthPlanProvider.from_checkpoint(
            tmp_path,
            device="cpu",
            dtype=torch.float32,
        )

    assert load_counts == {"processor": 0, "model": 0}


def test_from_checkpoint_rejects_non_integer_keyframe_offsets_before_loading(
    tmp_path,
    monkeypatch,
):
    module = load_provider_module()
    metadata = valid_metadata()
    metadata["keyframe_offsets"] = [8.0]
    write_complete_checkpoint_layout(tmp_path, metadata)
    load_counts = install_forbidden_checkpoint_loaders(monkeypatch)

    with pytest.raises(ValueError, match="keyframe_offsets"):
        module.FrozenDinoDepthPlanProvider.from_checkpoint(
            tmp_path,
            device="cpu",
            dtype=torch.float32,
        )

    assert load_counts == {"processor": 0, "model": 0}


@pytest.mark.parametrize(
    "missing_name",
    [
        "plan_head.pt",
        "depth_head.pt",
        "current_plan_head.pt",
        "current_depth_head.pt",
        "plan_token_embedding.pt",
        "planner_meta.json",
        "qwen3vl_lora_or_model",
        "processor",
    ],
)
def test_validate_checkpoint_files_names_every_missing_entry(
    tmp_path,
    missing_name,
):
    module = load_provider_module()
    required_files = (
        "plan_head.pt",
        "depth_head.pt",
        "current_plan_head.pt",
        "current_depth_head.pt",
        "plan_token_embedding.pt",
        "planner_meta.json",
    )
    required_dirs = ("qwen3vl_lora_or_model", "processor")
    for name in required_files:
        (tmp_path / name).touch()
    for name in required_dirs:
        (tmp_path / name).mkdir()

    missing_path = tmp_path / missing_name
    if missing_path.is_dir():
        missing_path.rmdir()
    else:
        missing_path.unlink()

    with pytest.raises(FileNotFoundError, match=missing_name):
        module.validate_checkpoint_files(tmp_path)


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
        return FakeBatch({"input_ids": torch.ones(len(images), 4, dtype=torch.long)})


class FakeWrapper:
    def __init__(self):
        self.calls = 0
        self.training = True
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))

    def eval(self):
        self.training = False
        return self

    def parameters(self):
        return iter((self.anchor,))

    def predict_dino_depth_plan(self, **inputs):
        self.calls += 1
        batch = inputs["input_ids"].shape[0]
        dino = torch.ones(batch, 256, 1024, requires_grad=True)
        depth = torch.full(
            (batch, 256, 1024),
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
        device=torch.device("cpu"),
        input_builder=processor.build_inputs,
    )
    images = torch.zeros(2, 3, 12, 20)
    result = provider.predict(images, ["open drawer", "pick mug"])

    assert wrapper.calls == 1
    assert wrapper.training is False
    assert result.dino_plan.shape == (2, 256, 1024)
    assert result.depth_plan.shape == (2, 256, 1024)
    assert result.semantic_plan_times.shape == (2, 1)
    assert result.dino_plan.requires_grad is False
    assert result.depth_plan.requires_grad is False
    assert processor.instructions == ["open drawer", "pick mug"]
    assert all(isinstance(image, Image.Image) for image in processor.images)


def test_from_components_freezes_every_wrapper_parameter():
    module = load_provider_module()
    processor = FakeProcessor()
    wrapper = FakeWrapper()

    provider = module.FrozenDinoDepthPlanProvider.from_components(
        processor=processor,
        wrapper=wrapper,
        contract=module.validate_planner_metadata(valid_metadata()),
        device="cpu",
        input_builder=processor.build_inputs,
    )

    assert not isinstance(provider, torch.nn.Module)
    assert wrapper.anchor.requires_grad is False


def test_predict_rejects_mismatched_instruction_count():
    module = load_provider_module()
    processor = FakeProcessor()
    provider = module.FrozenDinoDepthPlanProvider.from_components(
        processor=processor,
        wrapper=FakeWrapper(),
        contract=module.validate_planner_metadata(valid_metadata()),
        device="cpu",
        input_builder=processor.build_inputs,
    )
    with pytest.raises(ValueError, match="batch mismatch"):
        provider.predict(torch.zeros(2, 3, 8, 8), ["only one"])


def test_image_tensor_batch_rejects_non_finite_input():
    module = load_provider_module()
    images = torch.zeros(1, 3, 8, 8)
    images[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        module.image_tensor_batch_to_pil(images)


@pytest.mark.parametrize(
    "images",
    [torch.zeros(3, 8, 8), torch.zeros(1, 4, 8, 8)],
)
def test_image_tensor_batch_rejects_non_bchw_rgb_input(images):
    module = load_provider_module()
    with pytest.raises(ValueError, match="shape"):
        module.image_tensor_batch_to_pil(images)


def test_image_tensor_batch_rejects_values_outside_normalized_range():
    module = load_provider_module()
    images = torch.zeros(1, 3, 8, 8)
    images[0, 0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="normalized"):
        module.image_tensor_batch_to_pil(images)


class InvalidOutputWrapper(FakeWrapper):
    def __init__(self, *, invalid_shape=None, non_finite=None):
        super().__init__()
        self.invalid_shape = invalid_shape
        self.non_finite = non_finite

    def predict_dino_depth_plan(self, **inputs):
        batch = inputs["input_ids"].shape[0]
        dino = torch.zeros(batch, 256, 1024)
        depth = torch.zeros(batch, 256, 1024)
        if self.invalid_shape == "dino_plan":
            dino = torch.zeros(batch, 255, 1024)
        elif self.invalid_shape == "depth_plan":
            depth = torch.zeros(batch, 255, 1024)
        if self.non_finite == "dino_plan":
            dino[0, 0, 0] = float("nan")
        elif self.non_finite == "depth_plan":
            depth[0, 0, 0] = float("nan")
        return dino, depth


@pytest.mark.parametrize("branch", ["dino_plan", "depth_plan"])
def test_predict_rejects_wrong_branch_shape(branch):
    module = load_provider_module()
    processor = FakeProcessor()
    provider = module.FrozenDinoDepthPlanProvider.from_components(
        processor=processor,
        wrapper=InvalidOutputWrapper(invalid_shape=branch),
        contract=module.validate_planner_metadata(valid_metadata()),
        device="cpu",
        input_builder=processor.build_inputs,
    )

    with pytest.raises(RuntimeError, match=branch):
        provider.predict(torch.zeros(1, 3, 8, 8), ["pick mug"])


@pytest.mark.parametrize("branch", ["dino_plan", "depth_plan"])
def test_predict_rejects_non_finite_branch(branch):
    module = load_provider_module()
    processor = FakeProcessor()
    provider = module.FrozenDinoDepthPlanProvider.from_components(
        processor=processor,
        wrapper=InvalidOutputWrapper(non_finite=branch),
        contract=module.validate_planner_metadata(valid_metadata()),
        device="cpu",
        input_builder=processor.build_inputs,
    )

    with pytest.raises(RuntimeError, match=branch):
        provider.predict(torch.zeros(1, 3, 8, 8), ["pick mug"])


class TinyPlannerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=8),
            image_token_id=42,
        )

    def get_input_embeddings(self):
        return self.embedding


def tiny_export_metadata():
    return {
        "semantic_dim": 4,
        "plan_token_ids": [3, 4, 5, 6, 7, 8],
        "target_tokens": 2,
        "num_keyframes": 2,
        "grid_size": 1,
        "branch_latent_per_keyframe": 2,
        "shared_latent_per_keyframe": 1,
        "private_latent_per_keyframe": 1,
        "plan_head_type": "lingbot_dino",
        "plan_head_num_heads": 4,
        "plan_head_dropout": 0.0,
        "sem_mlp_hidden_size": 0,
        "mse_loss_weight": 1.0,
        "cosine_loss_weight": 0.0,
        "norm_loss_weight": 0.0,
        "variance_loss_weight": 0.0,
        "infonce_loss_weight": 0.0,
        "infonce_temperature": 0.07,
        "depth_feature_dim": 4,
        "depth_grid_size": 1,
        "depth_loss_weight": 0.004,
    }


def write_tiny_export(module, checkpoint):
    metadata = tiny_export_metadata()
    source = module.PlannerWrapper(
        model=TinyPlannerModel(),
        hidden_size=8,
        semantic_dim=4,
        plan_token_ids=metadata["plan_token_ids"],
        target_len=2,
        num_keyframes=2,
        grid_size=1,
        num_latent_per_keyframe=2,
        shared_latent_per_keyframe=1,
        private_latent_per_keyframe=1,
        plan_head_type="lingbot_dino",
        plan_head_num_heads=4,
        plan_head_dropout=0.0,
        sem_mlp_hidden_size=0,
        mse_loss_weight=1.0,
        cosine_loss_weight=0.0,
        norm_loss_weight=0.0,
        variance_loss_weight=0.0,
        infonce_loss_weight=0.0,
        infonce_temperature=0.07,
        use_depth=True,
        depth_dim=4,
        depth_grid_size=1,
        depth_loss_weight=0.004,
    )
    with torch.no_grad():
        for index, parameter in enumerate(source.plan_head.parameters()):
            parameter.fill_(0.01 * (index + 1))
        for index, parameter in enumerate(source.depth_head.parameters()):
            parameter.fill_(0.02 * (index + 1))
    torch.save(source.plan_head.state_dict(), checkpoint / "plan_head.pt")
    torch.save(source.depth_head.state_dict(), checkpoint / "depth_head.pt")
    plan_embedding = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    torch.save(plan_embedding, checkpoint / "plan_token_embedding.pt")
    return source, metadata, plan_embedding


def test_from_exported_checkpoint_restores_and_freezes_all_components(
    tmp_path,
    monkeypatch,
):
    module = load_trainer_module()
    source, metadata, plan_embedding = write_tiny_export(module, tmp_path)
    target_model = TinyPlannerModel()
    with torch.no_grad():
        target_model.embedding.weight.fill_(-7.0)

    original_torch_load = torch.load
    load_calls = []

    def recording_torch_load(*args, **kwargs):
        load_calls.append((Path(args[0]).name, kwargs.copy()))
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(module.torch, "load", recording_torch_load)

    wrapper = module.PlannerWrapper.from_exported_checkpoint(
        model=target_model,
        checkpoint_dir=tmp_path,
        metadata=metadata,
    )

    assert wrapper.training is False
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())
    assert wrapper.depth_loss_weight == pytest.approx(0.004)
    for name, expected in source.plan_head.state_dict().items():
        assert torch.equal(wrapper.plan_head.state_dict()[name], expected)
    for name, expected in source.depth_head.state_dict().items():
        assert torch.equal(wrapper.depth_head.state_dict()[name], expected)
    plan_ids = torch.tensor(metadata["plan_token_ids"], dtype=torch.long)
    assert torch.equal(
        target_model.embedding.weight[plan_ids],
        plan_embedding,
    )
    assert [name for name, _kwargs in load_calls] == [
        "plan_head.pt",
        "depth_head.pt",
        "plan_token_embedding.pt",
    ]
    assert all(
        kwargs == {"map_location": "cpu", "weights_only": True}
        for _name, kwargs in load_calls
    )


def test_from_exported_checkpoint_rejects_plan_embedding_shape(tmp_path):
    module = load_trainer_module()
    _source, metadata, _plan_embedding = write_tiny_export(module, tmp_path)
    torch.save(torch.zeros(5, 8), tmp_path / "plan_token_embedding.pt")

    with pytest.raises(ValueError, match="plan token embedding shape"):
        module.PlannerWrapper.from_exported_checkpoint(
            model=TinyPlannerModel(),
            checkpoint_dir=tmp_path,
            metadata=metadata,
        )


def test_from_exported_checkpoint_loads_head_state_strictly(tmp_path):
    module = load_trainer_module()
    source, metadata, _plan_embedding = write_tiny_export(module, tmp_path)
    incomplete_state = source.plan_head.state_dict()
    incomplete_state.pop(next(iter(incomplete_state)))
    torch.save(incomplete_state, tmp_path / "plan_head.pt")

    with pytest.raises(RuntimeError, match="Missing key"):
        module.PlannerWrapper.from_exported_checkpoint(
            model=TinyPlannerModel(),
            checkpoint_dir=tmp_path,
            metadata=metadata,
        )


def write_complete_checkpoint_layout(checkpoint, metadata):
    for name in (
        "plan_head.pt",
        "depth_head.pt",
        "current_plan_head.pt",
        "current_depth_head.pt",
        "plan_token_embedding.pt",
    ):
        (checkpoint / name).touch()
    for name in ("qwen3vl_lora_or_model", "processor"):
        (checkpoint / name).mkdir()
    (checkpoint / "planner_meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def install_forbidden_checkpoint_loaders(monkeypatch):
    load_counts = {"processor": 0, "model": 0}

    class ForbiddenProcessorLoader:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            load_counts["processor"] += 1
            raise AssertionError("processor loading must follow schema validation")

    class ForbiddenModelLoader:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            load_counts["model"] += 1
            raise AssertionError("model loading must follow schema validation")

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = ForbiddenProcessorLoader
    transformers.Qwen3VLForConditionalGeneration = ForbiddenModelLoader
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return load_counts


def install_checkpoint_loading_fakes(monkeypatch, *, vocab_size=512):
    state = SimpleNamespace(
        processor=FakeProcessor(),
        processor_load_calls=[],
        model_load_calls=[],
        wrapper_load_calls=[],
    )

    class CheckpointModel:
        def __init__(self):
            self.device = None
            self.embedding = torch.nn.Embedding(vocab_size, 8)

        def to(self, device):
            self.device = torch.device(device)
            return self

        def get_input_embeddings(self):
            return self.embedding

    state.model = CheckpointModel()

    class ProcessorLoader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state.processor_load_calls.append((Path(path), kwargs))
            return state.processor

    class ModelLoader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state.model_load_calls.append((Path(path), kwargs))
            return state.model

    class CheckpointWrapper(FakeWrapper):
        @classmethod
        def from_exported_checkpoint(
            cls,
            *,
            model,
            checkpoint_dir,
            metadata,
        ):
            consumed_metadata = {
                field: metadata[field] for field in RECONSTRUCTION_METADATA_FIELDS
            }
            wrapper = cls()
            wrapper.reconstruction_metadata = consumed_metadata
            state.wrapper_load_calls.append(
                (model, Path(checkpoint_dir), consumed_metadata)
            )
            return wrapper

        def to(self, device):
            self.device = torch.device(device)
            return self

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = ProcessorLoader
    transformers.Qwen3VLForConditionalGeneration = ModelLoader
    trainer = ModuleType("train_qwen3vl4b_lingbot_dino_planner")
    trainer.PlannerWrapper = CheckpointWrapper
    trainer.build_planner_inputs = (
        lambda processor, images, instructions, plan_sequence: (
            processor.build_inputs(images, instructions, plan_sequence)
        )
    )
    trainer.move_qwen_inputs_to_device = lambda inputs, device, model_dtype: inputs.to(
        device
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(
        sys.modules,
        "train_qwen3vl4b_lingbot_dino_planner",
        trainer,
    )
    return state


@pytest.mark.parametrize("field", RECONSTRUCTION_METADATA_FIELDS)
def test_from_checkpoint_rejects_missing_reconstruction_field_before_loading(
    tmp_path,
    monkeypatch,
    field,
):
    module = load_provider_module()
    metadata = valid_metadata()
    del metadata[field]
    write_complete_checkpoint_layout(tmp_path, metadata)
    load_counts = install_forbidden_checkpoint_loaders(monkeypatch)

    with pytest.raises(ValueError, match=field):
        module.FrozenDinoDepthPlanProvider.from_checkpoint(
            tmp_path,
            device="cpu",
            dtype=torch.float32,
        )

    assert load_counts == {"processor": 0, "model": 0}


@pytest.mark.parametrize("field", NUMERIC_METADATA_FIELDS)
@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param(None, id="null"),
        pytest.param(True, id="bool"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param("not-numeric", id="wrong-type"),
    ],
)
def test_from_checkpoint_rejects_malformed_numeric_field_before_loading(
    tmp_path,
    monkeypatch,
    field,
    invalid_value,
):
    module = load_provider_module()
    metadata = valid_metadata()
    metadata[field] = invalid_value
    write_complete_checkpoint_layout(tmp_path, metadata)
    load_counts = install_forbidden_checkpoint_loaders(monkeypatch)

    with pytest.raises(ValueError, match=field):
        module.FrozenDinoDepthPlanProvider.from_checkpoint(
            tmp_path,
            device="cpu",
            dtype=torch.float32,
        )

    assert load_counts == {"processor": 0, "model": 0}


def malformed_plan_token_ids():
    valid_ids = list(range(3, 19))
    duplicate_ids = valid_ids.copy()
    duplicate_ids[-1] = duplicate_ids[0]
    negative_ids = valid_ids.copy()
    negative_ids[0] = -1
    boolean_ids = valid_ids.copy()
    boolean_ids[0] = False
    float_ids = valid_ids.copy()
    float_ids[0] = 3.0
    nan_ids = valid_ids.copy()
    nan_ids[0] = float("nan")
    return [
        pytest.param(None, id="null"),
        pytest.param("not-a-list", id="wrong-container"),
        pytest.param(valid_ids[:-1], id="wrong-length"),
        pytest.param(duplicate_ids, id="duplicate"),
        pytest.param(negative_ids, id="negative"),
        pytest.param(boolean_ids, id="boolean"),
        pytest.param(float_ids, id="float"),
        pytest.param(nan_ids, id="non-finite"),
    ]


@pytest.mark.parametrize("invalid_ids", malformed_plan_token_ids())
def test_from_checkpoint_rejects_malformed_plan_token_ids_before_loading(
    tmp_path,
    monkeypatch,
    invalid_ids,
):
    module = load_provider_module()
    metadata = valid_metadata()
    metadata["plan_token_ids"] = invalid_ids
    write_complete_checkpoint_layout(tmp_path, metadata)
    load_counts = install_forbidden_checkpoint_loaders(monkeypatch)

    with pytest.raises(ValueError, match="plan_token_ids"):
        module.FrozenDinoDepthPlanProvider.from_checkpoint(
            tmp_path,
            device="cpu",
            dtype=torch.float32,
        )

    assert load_counts == {"processor": 0, "model": 0}


def test_from_checkpoint_validates_metadata_before_model_loading(
    tmp_path,
    monkeypatch,
):
    module = load_provider_module()
    metadata = valid_metadata()
    metadata["query_layout"] = "wrong_layout"
    write_complete_checkpoint_layout(tmp_path, metadata)
    load_counts = install_forbidden_checkpoint_loaders(monkeypatch)

    with pytest.raises(ValueError, match="query_layout"):
        module.FrozenDinoDepthPlanProvider.from_checkpoint(
            tmp_path,
            device="cpu",
            dtype=torch.float32,
        )
    assert load_counts == {"processor": 0, "model": 0}


def test_from_checkpoint_rejects_token_id_outside_model_vocabulary(
    tmp_path,
    monkeypatch,
):
    module = load_provider_module()
    metadata = valid_metadata()
    metadata["plan_token_ids"][-1] = 512
    write_complete_checkpoint_layout(tmp_path, metadata)
    state = install_checkpoint_loading_fakes(monkeypatch, vocab_size=512)

    with pytest.raises(ValueError, match="plan_token_ids.*vocabulary"):
        module.FrozenDinoDepthPlanProvider.from_checkpoint(
            tmp_path,
            device="cpu",
            dtype=torch.float32,
        )

    assert len(state.processor_load_calls) == 1
    assert len(state.model_load_calls) == 1
    assert state.wrapper_load_calls == []


def test_from_checkpoint_wires_local_frozen_components(
    tmp_path,
    monkeypatch,
):
    module = load_provider_module()
    metadata = valid_metadata()
    write_complete_checkpoint_layout(tmp_path, metadata)
    state = install_checkpoint_loading_fakes(monkeypatch)

    provider = module.FrozenDinoDepthPlanProvider.from_checkpoint(
        tmp_path,
        device="cpu",
        dtype=torch.float32,
    )
    result = provider.predict(
        torch.zeros(1, 3, 8, 8),
        ["pick mug"],
    )

    assert result.dino_plan.shape == (1, 256, 1024)
    assert state.processor_load_calls == [
        (tmp_path / "processor", {"local_files_only": True})
    ]
    assert state.model_load_calls == [
        (
            tmp_path / "qwen3vl_lora_or_model",
            {"torch_dtype": torch.float32, "local_files_only": True},
        )
    ]
    assert state.model.device == torch.device("cpu")
    assert len(state.wrapper_load_calls) == 1
    assert state.wrapper_load_calls[0][0] is state.model
    assert state.wrapper_load_calls[0][1] == tmp_path
    assert state.wrapper_load_calls[0][2] == {
        field: metadata[field] for field in RECONSTRUCTION_METADATA_FIELDS
    }
    assert provider.wrapper.reconstruction_metadata == state.wrapper_load_calls[0][2]
    assert provider.wrapper.training is False
    assert provider.wrapper.anchor.requires_grad is False
