# GE-Act Dual-Camera VLM Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-image Qwen3-VL planner that predicts ordered main/wrist future SigLIP2 grids, then use its frozen output as one-keyframe semantic guidance for GE-Act.

**Architecture:** A GE-Act dataset adapter exposes the current and endpoint frames as two independent camera images. One VLM forward produces shared planner-token states plus two separately gathered image-token spans; the existing four 64-token heads are shared and decoded once per view. A frozen GE-Act provider returns `[B,2,1,256,1024]`, which the existing view-aligned LTX semantic adapter consumes with one normalized future time.

**Tech Stack:** Python 3.10+, PyTorch, Transformers Qwen3-VL, SigLIP2, Depth Anything 3, Accelerate/DeepSpeed ZeRO-2, OmegaConf, GE-Act LTX, pytest.

## Global Constraints

- Camera order is exactly `main=0`, `wrist=1`.
- VLM input uses two independent image slots and never constructs a horizontal composite.
- The planner keeps four independent 64-token query groups, 256 total query tokens.
- Current/future SigLIP2 and current/future depth supervision remain enabled for both views.
- Per-view SigLIP2 and depth outputs are `[B,2,256,1024]`.
- The only exported future keyframe is offset `8` in a nine-frame future window, normalized to time `1.0`.
- `/data/users/junjie/code/VLM4WAM_k1_zero2_bidir/outputs/qwen3vl2b_siglip2_da3_libero_cur_k1/step_030000` is initialization only; legacy composite metadata is rejected by dual-camera inference.
- GE-Act freezes the planner and trains only LTX plus the semantic adapter during the initial integration.
- Production code is written only after its focused test has failed for the intended reason.

---

### Task 1: Independent GE-Act camera dataset and VLM input builder

**Files:**
- Create: `qwen3_vl_semantic_planner/ge_act_dual_camera.py`
- Create: `tests/test_ge_act_dual_camera_planner.py`

**Interfaces:**
- Consumes: a GE-Act sample with `video: Tensor[C,V,T,H,W]` in `[-1,1]` and `caption: str`.
- Produces: `GEActDualCameraPlannerDataset`, `build_dual_camera_planner_inputs`, and `DualCameraPlannerCollator`.
- Dataset item keys: `images`, `current_camera_images`, `future_camera_images`, `prompt`, and `stem`.

- [ ] **Step 1: Write failing tests for camera order, temporal selection, and two image slots**

```python
def test_ge_act_adapter_selects_current_and_future_endpoint_without_concat():
    video = torch.zeros(3, 2, 13, 2, 2)
    video[:, 0, 3].fill_(-0.5)
    video[:, 1, 3].fill_(0.0)
    video[:, 0, 12].fill_(0.5)
    video[:, 1, 12].fill_(1.0)
    wrapped = GEActDualCameraPlannerDataset(
        FakeDataset({"video": video, "caption": "pick the cup"}),
        n_previous=4,
        future_offset=8,
    )
    item = wrapped[0]
    assert item["current_camera_images"].shape == (2, 2, 2, 3)
    assert item["future_camera_images"].shape == (2, 2, 2, 3)
    assert item["images"][0].getpixel((0, 0))[0] == 64
    assert item["images"][1].getpixel((0, 0))[0] == 128
    assert item["prompt"] == "pick the cup"


def test_dual_camera_input_builder_flattens_images_main_then_wrist():
    processor = RecordingProcessor()
    main = Image.new("RGB", (2, 2), "red")
    wrist = Image.new("RGB", (2, 2), "blue")
    build_dual_camera_planner_inputs(
        processor,
        [(main, wrist)],
        ["pick"],
        ["<|sem_plan_0|>"],
    )
    assert processor.images == [main, wrist]
    assert processor.texts[0].count("<|image_pad|>") == 2
    assert "Main camera" in processor.rendered_conversations[0]
    assert "Wrist camera" in processor.rendered_conversations[0]
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py -k 'adapter or input_builder'`

Expected: collection fails because `qwen3_vl_semantic_planner.ge_act_dual_camera` does not exist.

- [ ] **Step 3: Implement the focused data contract**

```python
CAMERA_NAMES = ("main", "wrist")


def normalized_hwc_camera_frames_to_pil(frames: torch.Tensor) -> tuple[Image.Image, Image.Image]:
    if tuple(frames.shape[:1]) != (2,) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"camera frames must be [2,H,W,3], got {tuple(frames.shape)}")
    if not torch.isfinite(frames).all() or frames.min() < -1.0001 or frames.max() > 1.0001:
        raise ValueError("camera frames must be finite and normalized to [-1,1]")
    rgb = ((frames.float().cpu() + 1.0) * 127.5).round().clamp(0, 255).byte()
    return tuple(Image.fromarray(frame.numpy(), mode="RGB") for frame in rgb)


class GEActDualCameraPlannerDataset(Dataset):
    def __init__(self, dataset: Dataset, *, n_previous: int = 4, future_offset: int = 8):
        self.dataset = dataset
        self.n_previous = int(n_previous)
        self.future_offset = int(future_offset)

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, _epoch: int) -> None:
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        video = sample["video"]
        expected_future_index = self.n_previous + self.future_offset
        if video.ndim != 5 or video.shape[0] != 3 or video.shape[1] != 2:
            raise ValueError(f"GE-Act planner video must be [3,2,T,H,W], got {tuple(video.shape)}")
        if expected_future_index >= video.shape[2]:
            raise ValueError(f"future index {expected_future_index} exceeds T={video.shape[2]}")
        current = video[:, :, self.n_previous - 1].permute(1, 2, 3, 0).contiguous()
        future = video[:, :, expected_future_index].permute(1, 2, 3, 0).contiguous()
        return {
            "stem": f"geact_{index:09d}",
            "images": normalized_hwc_camera_frames_to_pil(current),
            "current_camera_images": current,
            "future_camera_images": future,
            "prompt": str(sample["caption"]),
        }
```

Implement `build_dual_camera_planner_inputs` with content ordered as text label, image slot, text label, image slot, instruction, then flatten image pairs in sample-major alternating order: main sample 0, wrist sample 0, main sample 1, wrist sample 1. Implement `DualCameraPlannerCollator` to stack current/future tensors as `[B,2,H,W,3]` and append them to the processor batch.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py -k 'adapter or input_builder'`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the dataset boundary**

```bash
git add qwen3_vl_semantic_planner/ge_act_dual_camera.py tests/test_ge_act_dual_camera_planner.py
git commit -m "feat(planner): add independent GE-Act camera inputs"
```

---

### Task 2: Per-view VLM image-hidden routing with shared 256 query tokens

**Files:**
- Modify: `qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py:1176-2145`
- Modify: `tests/test_ge_act_dual_camera_planner.py`

**Interfaces:**
- Consumes: `input_ids` containing exactly two contiguous image-token spans per batch item.
- Produces: `PlannerWrapper.predict_current_future_plans()` dictionaries whose values are `[B,2,256,D]` when `num_camera_views=2`.

- [ ] **Step 1: Add failing tests for span gathering and per-view head routing**

```python
def test_collect_image_hidden_keeps_two_contiguous_camera_spans():
    wrapper = PlannerWrapper.__new__(PlannerWrapper)
    wrapper.image_token_id = 99
    hidden = torch.arange(1 * 10 * 3).reshape(1, 10, 3).float()
    input_ids = torch.tensor([[1, 99, 99, 2, 3, 99, 99, 4, 5, 6]])
    actual = wrapper.collect_image_hidden_by_view(hidden, input_ids, num_views=2)
    assert actual.shape == (1, 2, 2, 3)
    torch.testing.assert_close(actual[0, 0], hidden[0, 1:3])
    torch.testing.assert_close(actual[0, 1], hidden[0, 5:7])


def test_dual_camera_wrapper_reuses_four_query_groups_for_both_views():
    wrapper = make_fake_dual_camera_wrapper(num_task_tokens=64)
    plans = wrapper.predict_current_future_plans(input_ids=torch.ones(2, 1, dtype=torch.long))
    assert wrapper.latent_len == 4 * 64
    assert set(plans) == {"current_dino", "future_dino", "current_depth", "future_depth"}
    assert all(value.shape == (2, 2, 256, 1024) for value in plans.values())
    assert not torch.equal(plans["future_dino"][:, 0], plans["future_dino"][:, 1])


def test_dual_camera_loss_detects_swapped_teacher_views():
    wrapper = make_loss_only_wrapper_with_unit_branch_weights()
    plans = make_four_branch_plans(main_value=0.0, wrist_value=10.0)
    aligned = make_four_branch_plans(main_value=0.0, wrist_value=10.0)
    swapped = {name: value.flip(1) for name, value in aligned.items()}
    aligned_loss = wrapper.compute_current_future_losses(plans, aligned)["loss"]
    swapped_loss = wrapper.compute_current_future_losses(plans, swapped)["loss"]
    assert aligned_loss == 0
    assert swapped_loss > 0
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py -k 'image_hidden or reuses_four'`

Expected: tests fail because `collect_image_hidden_by_view` and `num_camera_views` are absent.

- [ ] **Step 3: Add the minimal dual-view routing**

Add `num_camera_views: int = 1` to `PlannerWrapper.__init__`, validate it is `1` or `2`, and retain the existing one-view behavior. Add this span gatherer:

```python
def collect_image_hidden_by_view(self, hidden, input_ids, *, num_views):
    rows = []
    for batch_index in range(input_ids.shape[0]):
        positions = torch.nonzero(
            input_ids[batch_index] == int(self.image_token_id), as_tuple=False
        ).flatten()
        split_points = torch.nonzero(positions[1:] != positions[:-1] + 1, as_tuple=False).flatten() + 1
        spans = torch.tensor_split(positions, split_points.cpu().tolist())
        if len(spans) != num_views or any(span.numel() == 0 for span in spans):
            raise RuntimeError(f"expected {num_views} image-token spans, got {len(spans)}")
        if len({int(span.numel()) for span in spans}) != 1:
            raise RuntimeError("dual-camera image-token spans must have equal length")
        rows.append(torch.stack([hidden[batch_index, span] for span in spans], dim=0))
    return torch.stack(rows, dim=0)
```

In `_forward_hiddens`, return `[B,2,N,H]` image context when `num_camera_views=2`. In `predict_current_future_plans`, split the four task-hidden groups once, call each existing head once per camera span, and stack view outputs on dimension `1`. Do not create new heads or query banks.

Update `forward` to validate semantic target length using `semantic_plan_labels.shape[-2]`, so both `[B,256,D]` and `[B,2,256,D]` remain supported. Compute all four existing losses directly over the added view dimension.

- [ ] **Step 4: Run planner tests and existing single-view regressions**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py tests/test_lingbot_planner_evaluation.py`

Expected: all tests pass; the legacy one-view API still returns `[B,256,D]`.

- [ ] **Step 5: Commit per-view routing**

```bash
git add qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py tests/test_ge_act_dual_camera_planner.py
git commit -m "feat(planner): predict aligned features for two cameras"
```

---

### Task 3: Exact legacy initialization and strict dual-camera exports

**Files:**
- Modify: `qwen3_vl_semantic_planner/qwen3vl_wrapper.py:38-82`
- Modify: `qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py:1528-1650,2244-2625`
- Modify: `tests/test_ge_act_dual_camera_planner.py`

**Interfaces:**
- Produces: `load_planner_initialization(wrapper, checkpoint_dir)` and dual-camera metadata fields.
- `load_qwen3vl_model_and_processor` adds the keyword argument `processor_path: str | Path | None = None` so saved model and processor directories can differ.

- [ ] **Step 1: Add failing checkpoint tests**

```python
def test_legacy_checkpoint_initializes_four_shared_heads_without_expansion(tmp_path):
    source = make_four_head_checkpoint(tmp_path, query_tokens=64)
    wrapper = make_fake_dual_camera_wrapper(num_task_tokens=64)
    report = load_planner_initialization(wrapper, source)
    assert report["loaded_heads"] == [
        "plan_head", "depth_head", "current_plan_head", "current_depth_head"
    ]
    assert wrapper.plan_head.query_embs.shape == (64 * 256, wrapper.plan_head.query_embs.shape[1])
    assert wrapper.latent_len == 256


def test_dual_camera_metadata_rejects_legacy_composite_inference():
    metadata = valid_dual_camera_metadata()
    metadata["planner_input_layout"] = "fastwam_current_multicamera_composite"
    with pytest.raises(ValueError, match="separate_camera_images"):
        validate_dual_camera_export_metadata(metadata)
```

- [ ] **Step 2: Run checkpoint tests and verify RED**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py -k 'checkpoint or metadata'`

Expected: tests fail because initialization and dual metadata validators are absent.

- [ ] **Step 3: Implement strict initialization and metadata**

Add `--init-planner-checkpoint`. When present, load the model from
`qwen3vl_lora_or_model`, the processor from `processor`, and strict-load:

```python
HEAD_FILES = {
    "plan_head": "plan_head.pt",
    "depth_head": "depth_head.pt",
    "current_plan_head": "current_plan_head.pt",
    "current_depth_head": "current_depth_head.pt",
}


def load_planner_initialization(wrapper, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    loaded = []
    for attribute, filename in HEAD_FILES.items():
        head = getattr(wrapper, attribute)
        state = torch.load(checkpoint_dir / filename, map_location="cpu", weights_only=True)
        head.load_state_dict(state, strict=True)
        loaded.append(attribute)
    return {"loaded_heads": loaded, "source": str(checkpoint_dir)}
```

The saved Qwen model already contains the trained token embedding table; the four strict head loads restore the pooled query banks. Reject initialization when metadata does not describe four independent 64-token groups or when any tensor shape differs.

For a two-view run, export and validate:

```python
{
    "planner_input_layout": "separate_camera_images",
    "camera_names": ["main", "wrist"],
    "num_camera_views": 2,
    "camera_head_sharing": "shared_head_per_view_image_context",
    "semantic_output_layout": "batch_view_token_feature",
    "semantic_teacher": "siglip2-large-patch16-256",
    "future_keyframe_offsets": [8],
}
```

- [ ] **Step 4: Run focused and compatibility tests**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py tests/test_lingbot_planner_evaluation.py`

Expected: all tests pass, including strict rejection of composite inference metadata.

- [ ] **Step 5: Commit checkpoint support**

```bash
git add qwen3_vl_semantic_planner/qwen3vl_wrapper.py qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py tests/test_ge_act_dual_camera_planner.py
git commit -m "feat(planner): initialize and export dual-camera checkpoints"
```

---

### Task 4: GE-Act-native planner training path and online two-view teachers

**Files:**
- Modify: `qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py:160-410,2630-3090`
- Modify: `qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh`
- Create: `qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_siglip2da3.sh`
- Modify: `tests/test_ge_act_dual_camera_planner.py`

**Interfaces:**
- New CLI: `--ge-act-data-config PATH`, mutually exclusive with legacy dataset sources.
- Training targets: four tensors shaped `[B,2,256,1024]`.

- [ ] **Step 1: Add failing tests for GE-Act dataset selection and teacher reshape**

```python
def test_flatten_two_camera_frames_for_online_teachers_preserves_order():
    frames = torch.zeros(2, 2, 4, 4, 3, dtype=torch.uint8)
    frames[:, 0].fill_(10)
    frames[:, 1].fill_(20)
    flat = flatten_camera_teacher_frames(frames)
    assert flat.shape == (4, 3, 4, 4)
    assert flat[0].float().mean() == 10
    assert flat[1].float().mean() == 20
    assert flat[2].float().mean() == 10
    assert flat[3].float().mean() == 20


def test_teacher_features_restore_batch_view_layout():
    encoded = torch.arange(4 * 256 * 1024).reshape(4, 256, 1024)
    restored = restore_camera_teacher_features(encoded, batch_size=2, num_views=2)
    assert restored.shape == (2, 2, 256, 1024)
    torch.testing.assert_close(restored[0, 1], encoded[1])
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py -k 'teacher or dataset_selection'`

Expected: tests fail because the reshape helpers and GE-Act CLI source are missing.

- [ ] **Step 3: Implement the GE-Act training source**

Load `data.train` and the configured GE-Act dataset class from the YAML file, wrap it with `GEActDualCameraPlannerDataset(n_previous=4, future_offset=8)`, and use `DualCameraPlannerCollator`. Enforce exactly one of `dataset_root`, `fastwam_data_config`, or `ge_act_data_config`; a dual-camera run sets `num_camera_views=2`.

In the training loop:

```python
current = batch.pop("current_camera_images")  # [B,2,H,W,3]
future = batch.pop("future_camera_images")    # [B,2,H,W,3]
b, views = current.shape[:2]
current_bv = current.permute(0, 1, 4, 2, 3).reshape(b * views, 3, *current.shape[2:4])
future_bv = future.permute(0, 1, 4, 2, 3).reshape(b * views, 3, *future.shape[2:4])
current_siglip, future_siglip = dino_encoder.encode_current_and_future(current_bv, future_bv)
current_depth, future_depth = depth_encoder.encode_current_and_future(current_bv, future_bv)
batch["current_dino_labels"] = current_siglip.reshape(b, views, 256, 1024)
batch["semantic_plan_labels"] = future_siglip.reshape(b, views, 256, 1024)
batch["current_depth_labels"] = current_depth.reshape(b, views, 256, 1024)
batch["depth_plan_labels"] = future_depth.reshape(b, views, 256, 1024)
```

Update the shell launcher to accept `GE_ACT_DATA_CONFIG` and `INIT_PLANNER_CHECKPOINT`, set `NUM_TASK_TOKENS=64`, and pass the two new CLI arguments. The dedicated launcher records SigLIP2-large-patch16-256, DA3 last-layer, offset 8, full fine-tuning, and 30,000 steps.

- [ ] **Step 4: Run planner tests and shell syntax checks**

Run: `pytest -q tests/test_ge_act_dual_camera_planner.py`

Run: `bash -n qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_siglip2da3.sh`

Expected: tests pass and both shell scripts exit `0` from syntax checking.

- [ ] **Step 5: Commit the train path**

```bash
git add qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_siglip2da3.sh tests/test_ge_act_dual_camera_planner.py
git commit -m "feat(planner): train dual-camera targets from GE-Act data"
```

---

### Task 5: Frozen dual-camera planner provider for GE-Act

**Files:**
- Create: `ge_act/models/ltx_models/vlm_semantic_planner.py`
- Create: `tests/test_ge_act_vlm_semantic_planner.py`

**Interfaces:**
- Produces: `FrozenDualCameraVLMPlanner.from_checkpoint(checkpoint_dir, *, device, dtype)`.
- `predict(current_images, instructions)` consumes `[B,2,3,H,W]` in `[-1,1]` and returns `DualCameraSemanticPlan`.
- `DualCameraSemanticPlan.semantic_tokens` is `[B,2,1,256,1024]`; `times` is `[B*2,1]` filled with `1.0`.

- [ ] **Step 1: Add failing provider contract tests**

```python
def test_frozen_provider_returns_one_ordered_future_grid_per_camera():
    provider = FrozenDualCameraVLMPlanner.from_components(
        wrapper=FakeDualWrapper(),
        processor=FakeProcessor(),
        input_builder=fake_input_builder,
        input_mover=lambda value: value,
        device="cpu",
    )
    plan = provider.predict(torch.zeros(2, 2, 3, 8, 8), ["a", "b"])
    assert plan.semantic_tokens.shape == (2, 2, 1, 256, 1024)
    assert plan.times.shape == (4, 1)
    torch.testing.assert_close(plan.times, torch.ones(4, 1))
    assert all(not parameter.requires_grad for parameter in provider.wrapper.parameters())


def test_provider_rejects_composite_checkpoint_metadata():
    metadata = valid_dual_camera_metadata()
    metadata["num_camera_views"] = 1
    with pytest.raises(ValueError, match="num_camera_views"):
        validate_dual_camera_planner_metadata(metadata)
```

- [ ] **Step 2: Run provider tests and verify RED**

Run: `pytest -q tests/test_ge_act_vlm_semantic_planner.py`

Expected: collection fails because the provider module does not exist.

- [ ] **Step 3: Implement the frozen provider**

```python
@dataclass(frozen=True)
class DualCameraSemanticPlan:
    semantic_tokens: torch.Tensor
    times: torch.Tensor


@torch.no_grad()
def predict(self, current_images, instructions):
    if tuple(current_images.shape[:3]) != (len(instructions), 2, 3):
        raise ValueError("current_images must be [B,2,3,H,W]")
    image_pairs = normalized_bvchw_to_pil_pairs(current_images)
    model_inputs = self.input_mover(
        self.input_builder(self.processor, image_pairs, instructions, self.plan_tokens)
    )
    plans = self.wrapper.predict_current_future_plans(**model_inputs)
    future = plans["future_dino"]
    expected = (current_images.shape[0], 2, 256, 1024)
    if tuple(future.shape) != expected or not torch.isfinite(future).all():
        raise RuntimeError(f"future_siglip must be finite with shape {expected}")
    return DualCameraSemanticPlan(
        semantic_tokens=future.detach().unsqueeze(2),
        times=torch.ones(current_images.shape[0] * 2, 1, device=future.device),
    )
```

`from_checkpoint` validates files and dual metadata before allocating Qwen, loads the model/processor locally, reconstructs `PlannerWrapper(num_camera_views=2)`, sets evaluation mode, and freezes every parameter.

- [ ] **Step 4: Run provider and existing semantic adapter tests**

Run: `pytest -q tests/test_ge_act_vlm_semantic_planner.py tests/test_ge_act_ltx_semantic_guidance.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the provider**

```bash
git add ge_act/models/ltx_models/vlm_semantic_planner.py tests/test_ge_act_vlm_semantic_planner.py
git commit -m "feat(ge-act): add frozen dual-camera VLM planner"
```

---

### Task 6: Select the VLM planner in GE-Act training and validation

**Files:**
- Modify: `ge_act/runner/ge_trainer.py:225-510,650-715,990-1045`
- Modify: `ge_act/main.py:6-40`
- Modify: `ge_act/scripts/preflight_ltx_siglip2.py`
- Create: `ge_act/configs/ltx_model/libero/video_model_libero_vlm_planner.yaml`
- Create: `ge_act/scripts/train_ltx_vlm_planner.sh`
- Create: `ge_act/scripts/sbatch_train_ltx_vlm_planner_hpc3.sh`
- Modify: `tests/test_ge_act_siglip2_config.py`
- Modify: `tests/test_ge_act_semantic_training_contract.py`

**Interfaces:**
- `semantic_plan.source` is either `gt_siglip2` or `vlm_planner`.
- VLM mode reads `planner_checkpoint`, uses current observation index `n_previous - 1`, and emits one keyframe.

- [ ] **Step 1: Add failing trainer and config tests**

```python
def test_vlm_planner_config_uses_one_endpoint_keyframe_and_global_batch_128():
    config = yaml.safe_load(VLM_CONFIG_PATH.read_text())
    semantic = config["semantic_plan"]
    assert semantic["source"] == "vlm_planner"
    assert semantic["keyframe_indices"] == [8]
    assert semantic["validation_mode"] == "planner"
    assert config["diffusion_model"]["config"]["semantic_plan_num_keyframes"] == 1
    assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128


def test_build_vlm_semantic_condition_uses_current_observation_only():
    video = torch.arange(2 * 3 * 2 * 13 * 2 * 2).reshape(2, 3, 2, 13, 2, 2).float()
    provider = RecordingPlannerProvider()
    tokens, times = build_vlm_semantic_condition(
        provider, video, ["a", "b"], n_previous=4
    )
    torch.testing.assert_close(provider.images, video[:, :, :, 3].permute(0, 2, 1, 3, 4))
    assert tokens.shape == (2, 2, 1, 256, 1024)
    assert times.shape == (4, 1)


def test_main_exposes_a_bounded_smoke_step_override():
    source = (GE_ACT_ROOT / "main.py").read_text()
    assert "--max_train_steps" in source
    assert "runner.args.train_steps = args.max_train_steps" in source
```

- [ ] **Step 2: Run trainer/config tests and verify RED**

Run: `pytest -q tests/test_ge_act_siglip2_config.py tests/test_ge_act_semantic_training_contract.py -k 'vlm or planner'`

Expected: tests fail because the VLM config and trainer helper are absent.

- [ ] **Step 3: Implement source selection and planner conditioning**

Initialize exactly one semantic source:

```python
source = semantic_config.get("source", "gt_siglip2")
if source == "gt_siglip2":
    self.semantic_encoder = OnlineSiglip2SemanticEncoder(
        semantic_config["model_name_or_path"],
        device=device,
        dtype=dtype,
        frame_microbatch_size=int(semantic_config.get("frame_microbatch_size", 16)),
        expected_tokens=int(semantic_config.get("tokens_per_frame", 256)),
        expected_feature_dim=int(semantic_config.get("feature_dim", 1024)),
    )
elif source == "vlm_planner":
    self.semantic_planner = FrozenDualCameraVLMPlanner.from_checkpoint(
        semantic_config["planner_checkpoint"],
        device=device,
        dtype=dtype,
    )
else:
    raise ValueError(f"unknown semantic_plan.source: {source}")
```

Add a pure `build_vlm_semantic_condition` helper that selects `video[:,:,:,n_previous-1]`, reorders it to `[B,2,3,H,W]`, calls the frozen provider, and returns its tokens/times. In training and `validation_mode=planner`, use this helper instead of future ground-truth frames. Keep semantic dropout unchanged and keep the provider outside optimizer and accelerator wrapping.

Add `--max_train_steps` as a positive optional integer in `ge_act/main.py`. Immediately after constructing `Trainer`, assign `runner.args.train_steps = args.max_train_steps` when it is provided. `train_ltx_vlm_planner.sh` passes `${MAX_TRAIN_STEPS}` only when the environment variable is non-empty, so a smoke run cannot accidentally inherit 30,000 steps.

Create a new YAML derived from the current LTX semantic config with:

```yaml
batch_size: 4
gradient_accumulation_steps: 4
semantic_plan:
  enabled: true
  source: vlm_planner
  planner_checkpoint: /data/user/jhe724/junjie/outputs/qwen3vl2b_siglip2_da3_libero_dual_camera/step_030000
  tokens_per_frame: 256
  feature_dim: 1024
  keyframe_indices: [8]
  dropout: 0.15
  validation_mode: planner
diffusion_model:
  config:
    semantic_plan_num_keyframes: 1
    semantic_plan_num_views: 2
```

Keep the LTX learning rate `2e-5`, semantic adapter rate `1e-4`, 30,000 steps, warmup 1,000, gradient checkpointing enabled, and global batch 128. Update preflight so `vlm_planner` requires the dual checkpoint directory and `[8]`, while `gt_siglip2` retains `[0,3,5,8]` and the SigLIP2 model requirement.

- [ ] **Step 4: Run GE-Act semantic tests and shell syntax checks**

Run: `pytest -q tests/test_ge_act_siglip2_config.py tests/test_ge_act_semantic_training_contract.py tests/test_ge_act_semantic_pipeline.py tests/test_ge_act_ltx_semantic_guidance.py tests/test_ge_act_vlm_semantic_planner.py`

Run: `bash -n ge_act/scripts/train_ltx_vlm_planner.sh ge_act/scripts/sbatch_train_ltx_vlm_planner_hpc3.sh`

Expected: all tests pass and both launchers pass syntax validation.

- [ ] **Step 5: Commit GE-Act integration**

```bash
git add ge_act/main.py ge_act/runner/ge_trainer.py ge_act/scripts/preflight_ltx_siglip2.py ge_act/configs/ltx_model/libero/video_model_libero_vlm_planner.yaml ge_act/scripts/train_ltx_vlm_planner.sh ge_act/scripts/sbatch_train_ltx_vlm_planner_hpc3.sh tests/test_ge_act_siglip2_config.py tests/test_ge_act_semantic_training_contract.py
git commit -m "feat(ge-act): train LTX with dual-camera VLM guidance"
```

---

### Task 7: Full verification and staged GPU smoke tests

**Files:**
- Modify only if a test exposes a defect in files from Tasks 1-6.

**Interfaces:**
- Produces: a verified dual-camera planner training command and a verified frozen-planner GE-Act command.

- [ ] **Step 1: Run the complete focused local test set**

Run:

```bash
pytest -q \
  tests/test_ge_act_dual_camera_planner.py \
  tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_lingbot_planner_evaluation.py \
  tests/test_ge_act_ltx_semantic_guidance.py \
  tests/test_ge_act_semantic_pipeline.py \
  tests/test_ge_act_semantic_training_contract.py \
  tests/test_ge_act_siglip2_config.py
```

Expected: all tests pass with no warnings introduced by the changed modules.

- [ ] **Step 2: Run static verification**

Run: `python -m compileall -q qwen3_vl_semantic_planner ge_act/models/ltx_models ge_act/runner`

Run: `git diff --check`

Expected: both commands exit `0`.

- [ ] **Step 3: Run a two-step dual-camera planner smoke test on the checkpoint host**

Sync only tracked source, then run:

```bash
RUN_KIND=smoke \
NUM_GPUS=1 BATCH_SIZE=1 GRAD_ACCUM=1 EXPECTED_GLOBAL_BATCH=1 \
MAX_STEPS=2 SAVE_STEPS=2 SAVE_START_STEP=0 \
INIT_PLANNER_CHECKPOINT=/data/users/junjie/code/VLM4WAM_k1_zero2_bidir/outputs/qwen3vl2b_siglip2_da3_libero_cur_k1/step_030000 \
bash qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_siglip2da3.sh
```

Expected: two optimizer steps complete, all four losses are finite, and exported metadata contains `planner_input_layout=separate_camera_images`, `num_camera_views=2`, and `num_task_tokens=64`.

- [ ] **Step 4: Run a one-step GE-Act smoke test with the dual checkpoint**

Point `planner_checkpoint` in a temporary config copy at the smoke checkpoint and run:

```bash
NPROC_PER_NODE=1 MAX_TRAIN_STEPS=1 \
bash ge_act/scripts/train_ltx_vlm_planner.sh \
ge_act/configs/ltx_model/libero/video_model_libero_vlm_planner.yaml
```

Expected: the log reports semantic plan shape `[B,2,1,256,1024]`, planner parameters remain frozen, one LTX backward/optimizer step completes, and peak memory is recorded.

- [ ] **Step 5: Review the final diff and commit smoke-only fixes**

Run: `git status --short`

Run: `git diff --check`

If smoke testing required tracked fixes, stage only the implementation and test files from this plan, then commit with:

```bash
git add qwen3_vl_semantic_planner/ge_act_dual_camera.py qwen3_vl_semantic_planner/qwen3vl_wrapper.py qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh qwen3_vl_semantic_planner/dinov3_da3_2b/train_ge_act_dual_camera_siglip2da3.sh ge_act/main.py ge_act/models/ltx_models/vlm_semantic_planner.py ge_act/runner/ge_trainer.py ge_act/scripts/preflight_ltx_siglip2.py ge_act/configs/ltx_model/libero/video_model_libero_vlm_planner.yaml ge_act/scripts/train_ltx_vlm_planner.sh ge_act/scripts/sbatch_train_ltx_vlm_planner_hpc3.sh tests/test_ge_act_dual_camera_planner.py tests/test_ge_act_vlm_semantic_planner.py tests/test_ge_act_siglip2_config.py tests/test_ge_act_semantic_training_contract.py
git commit -m "fix: pass dual-camera planner smoke tests"
```

Expected: no unrelated artifacts, logs, generated checkpoints, or temporary configs are staged.
