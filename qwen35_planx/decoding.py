"""Fail-closed visual-only KV-cache decoding for grounded future plans."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from qwen35_planx.config import CAMERA_NAMES, PlanGeometry
from qwen35_planx.instruction import format_grounded_prompt, parse_libero_instruction
from qwen35_planx.vocabulary import (
    CAMERA_TOKENS,
    FRAME_END_TOKENS,
    FRAME_START_TOKENS,
    PLAN_END_TOKEN,
    PLAN_START_TOKEN,
    STRUCTURE_TOKENS,
    VisualVocabularyLayout,
)


_GEOMETRY = PlanGeometry()
_NUM_KEYFRAMES = _GEOMETRY.num_keyframes
_TOKENS_PER_FRAME = _GEOMETRY.tokens_per_frame
_VISUAL_VOCAB_SIZE = _GEOMETRY.visual_vocab_size
_NUM_ROLES = 3


def _require_floating_tensor(
    name: str,
    value: Tensor,
    shape: tuple[int | None, ...],
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be floating-point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class GeneratedGroundedPlan:
    """Generated visual codes and grounded post-code features."""

    codes: Tensor
    code_embeddings: Tensor
    post_hidden: Tensor
    predicted_phrase_embeddings: Tensor
    semantic_features: Tensor
    relevance: Tensor
    fusion_gate: Tensor
    times: Tensor
    _enforce_released_geometry: bool = field(
        default=True,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self._enforce_released_geometry) is not bool:
            raise TypeError("_enforce_released_geometry must be boolean")
        if not isinstance(self.codes, Tensor):
            raise TypeError("codes must be a tensor")
        if self.codes.ndim not in (3, 4):
            raise ValueError("codes must have flat or dual-camera plan shape")
        if self.codes.ndim == 4 and self.codes.shape[1] != len(CAMERA_NAMES):
            raise ValueError("unflattened codes must have two camera rows")
        if tuple(self.codes.shape[-2:]) != (
            _NUM_KEYFRAMES,
            _TOKENS_PER_FRAME,
        ):
            raise ValueError("codes must contain exactly four 27x27 grids")
        if self.codes.dtype == torch.bool or self.codes.dtype.is_floating_point:
            raise TypeError("codes must contain integer visual IDs")
        if bool((self.codes < 0).any()) or bool(
            (self.codes >= _VISUAL_VOCAB_SIZE).any()
        ):
            raise ValueError("codes contain illegal visual IDs")

        prefix = tuple(self.codes.shape[:-2])
        frames = (_NUM_KEYFRAMES, _TOKENS_PER_FRAME)
        _require_floating_tensor(
            "code_embeddings",
            self.code_embeddings,
            (*prefix, *frames, None),
        )
        _require_floating_tensor(
            "post_hidden",
            self.post_hidden,
            (*prefix, *frames, None),
        )
        _require_floating_tensor(
            "predicted_phrase_embeddings",
            self.predicted_phrase_embeddings,
            (*prefix, _NUM_ROLES, None),
        )
        text_dim = int(self.predicted_phrase_embeddings.shape[-1])
        _require_floating_tensor(
            "semantic_features",
            self.semantic_features,
            (*prefix, *frames, text_dim),
        )
        _require_floating_tensor(
            "relevance",
            self.relevance,
            (*prefix, _NUM_KEYFRAMES, _NUM_ROLES, _TOKENS_PER_FRAME),
        )
        _require_floating_tensor(
            "fusion_gate",
            self.fusion_gate,
            (*prefix, *frames, 1),
        )
        _require_floating_tensor("times", self.times, (_NUM_KEYFRAMES,))
        if self._enforce_released_geometry and (
            self.code_embeddings.shape[-1],
            self.post_hidden.shape[-1],
            text_dim,
        ) != (
            _GEOMETRY.ta_code_dim,
            _GEOMETRY.qwen_hidden_dim,
            _GEOMETRY.text_align_dim,
        ):
            raise ValueError(
                "released generated plans require code/hidden/text widths "
                "1536/2048/1152"
            )
        if bool((self.relevance < 0).any()):
            raise ValueError("relevance must be non-negative")
        if not torch.allclose(
            self.relevance.sum(dim=-1),
            torch.ones_like(self.relevance[..., 0]),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("relevance must be normalized over each 27x27 grid")
        if bool((self.fusion_gate < 0).any()) or bool(
            (self.fusion_gate > 1).any()
        ):
            raise ValueError("fusion_gate must be in [0,1]")

    @classmethod
    def _from_test_components(
        cls,
        **values: Tensor,
    ) -> GeneratedGroundedPlan:
        """Construct a reduced-width plan for unit tests only."""

        return cls(**values, _enforce_released_geometry=False)


def unflatten_generated_plan(
    plan: GeneratedGroundedPlan,
    batch_size: int,
) -> GeneratedGroundedPlan:
    """Split a flat sample-major camera batch into explicit ``[B,2,...]``."""

    if not isinstance(plan, GeneratedGroundedPlan):
        raise TypeError("plan must be a GeneratedGroundedPlan")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if plan.codes.ndim != 3:
        raise ValueError("plan must have a flat leading camera dimension")
    camera_batch = batch_size * len(CAMERA_NAMES)
    names = (
        "codes",
        "code_embeddings",
        "post_hidden",
        "predicted_phrase_embeddings",
        "semantic_features",
        "relevance",
        "fusion_gate",
    )
    updates: dict[str, Tensor] = {}
    for name in names:
        value = getattr(plan, name)
        if value.shape[0] != camera_batch:
            raise ValueError(f"{name} leading dimension must equal batch_size*2")
        updates[name] = value.reshape(batch_size, 2, *value.shape[1:])
    return replace(plan, **updates)


def _validate_layout(
    layout: VisualVocabularyLayout,
    *,
    processor: Any,
) -> None:
    if not isinstance(layout, VisualVocabularyLayout):
        raise TypeError("layout must be a VisualVocabularyLayout")
    if layout.visual_end_id - layout.visual_start_id != _VISUAL_VOCAB_SIZE:
        raise ValueError("layout must expose exactly 65,536 visual token IDs")
    token_pairs = tuple(layout.structure_token_ids)
    if tuple(token for token, _ in token_pairs) != STRUCTURE_TOKENS:
        raise ValueError("layout structural tokens are malformed")
    structure_ids = tuple(token_id for _, token_id in token_pairs)
    if len(set(structure_ids)) != len(structure_ids):
        raise ValueError("layout structural token IDs must be unique")
    if any(
        layout.visual_start_id <= token_id < layout.visual_end_id
        for token_id in structure_ids
    ):
        raise ValueError("structural token IDs must be outside the visual range")

    tokenizer = getattr(processor, "tokenizer", None) or layout._tokenizer
    eos_value = getattr(tokenizer, "eos_token_id", None)
    eos_ids: tuple[int, ...]
    if eos_value is None:
        eos_ids = ()
    elif isinstance(eos_value, Sequence) and not isinstance(eos_value, (str, bytes)):
        eos_ids = tuple(int(value) for value in eos_value)
    else:
        eos_ids = (int(eos_value),)
    if any(
        layout.visual_start_id <= token_id < layout.visual_end_id
        for token_id in eos_ids
    ):
        raise ValueError("EOS must not overlap the visual vocabulary")
    if set(eos_ids).intersection(structure_ids):
        raise ValueError("EOS must not overlap structural delimiters")


def _resolve_components(
    planner: nn.Module,
    *,
    layout: VisualVocabularyLayout | None,
    processor: Any | None,
) -> tuple[VisualVocabularyLayout, Any]:
    if not isinstance(planner, nn.Module):
        raise TypeError("planner must be a torch module")
    if layout is None:
        layout = getattr(planner, "layout", None)
    if processor is None:
        processor = getattr(planner, "processor", None)
    if processor is None:
        collator = getattr(planner, "collator", None)
        processor = getattr(collator, "processor", None)
    if processor is None:
        raise ValueError("a Qwen multimodal processor is required for generation")
    _validate_layout(layout, processor=processor)  # type: ignore[arg-type]
    return layout, processor  # type: ignore[return-value]


def _language_backbone(planner: nn.Module) -> nn.Module:
    resolver = getattr(planner, "_language_backbone", None)
    if callable(resolver):
        backbone = resolver()
    else:
        candidate = getattr(planner, "backbone", None)
        base_model = getattr(candidate, "model", None)
        backbone = base_model if isinstance(base_model, nn.Module) else candidate
    if not isinstance(backbone, nn.Module):
        raise TypeError("planner must expose the Task 7 Qwen base model")
    return backbone


def _current_visual_embedding_rows(
    planner: nn.Module,
    layout: VisualVocabularyLayout,
) -> Tensor:
    """Resolve visual rows from the backbone's current embedding storage."""

    backbone = _language_backbone(planner)
    get_input_embeddings = getattr(backbone, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if isinstance(weight, Tensor):
            if (
                weight.ndim != 2
                or weight.shape[0] < layout.visual_end_id
                or layout.visual_start_id < 0
            ):
                raise ValueError(
                    "current Qwen input embedding does not contain the visual range"
                )
            rows = weight[layout.visual_start_id : layout.visual_end_id]
            if rows.shape[0] != _VISUAL_VOCAB_SIZE:
                raise ValueError("current Qwen visual slice must contain 65,536 rows")
            return rows
    rows = getattr(planner, "visual_embedding_weight", None)
    if (
        not isinstance(rows, Tensor)
        or rows.ndim != 2
        or rows.shape[0] != _VISUAL_VOCAB_SIZE
    ):
        raise ValueError(
            "planner must expose current Qwen visual embedding rows"
        )
    return rows


def _last_hidden(planner: nn.Module, output: Any) -> Tensor:
    resolver = getattr(planner, "_last_hidden", None)
    if callable(resolver):
        hidden = resolver(output)
    elif isinstance(output, Mapping):
        hidden = output.get("last_hidden_state")
    else:
        hidden = getattr(output, "last_hidden_state", None)
    if not isinstance(hidden, Tensor) or hidden.ndim != 3:
        raise TypeError("Qwen base model must return three-dimensional hidden states")
    return hidden


def _past_key_values(output: Any) -> Any:
    if isinstance(output, Mapping):
        past = output.get("past_key_values")
    else:
        past = getattr(output, "past_key_values", None)
    if past is None:
        raise RuntimeError("Qwen base model did not return a KV cache")
    return past


def _rope_deltas(
    output: Any,
    *,
    backbone: nn.Module,
    device: torch.device,
) -> Tensor:
    if isinstance(output, Mapping):
        value = output.get("rope_deltas")
    else:
        value = getattr(output, "rope_deltas", None)
    if value is None:
        value = getattr(backbone, "rope_deltas", None)
    if value is None:
        raise RuntimeError("Qwen multimodal prefill did not return rope_deltas")
    if (
        not isinstance(value, Tensor)
        or value.dtype == torch.bool
        or value.dtype.is_floating_point
        or value.numel() != 1
    ):
        raise RuntimeError("Qwen rope_deltas must contain one integer camera value")
    return value.to(device=device, dtype=torch.long).reshape(1, 1)


def _build_prompt_inputs(
    *,
    processor: Any,
    layout: VisualVocabularyLayout,
    image: Tensor,
    instruction: str,
    camera: str,
    device: torch.device,
) -> dict[str, Tensor]:
    fields = parse_libero_instruction(instruction)
    camera_token = CAMERA_TOKENS[CAMERA_NAMES.index(camera)]
    text = f"{camera_token}\n{format_grounded_prompt(fields)}"
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if callable(apply_chat_template):
        text = apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    processed = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=False,
    )
    if not isinstance(processed, Mapping) or "input_ids" not in processed:
        raise ValueError("Qwen processor must return a mapping with input_ids")
    inputs: dict[str, Tensor] = {}
    for name, value in processed.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"Qwen processor field {name} must be a tensor")
        inputs[str(name)] = value.to(device=device)
    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Qwen processor must return one nonempty prompt sequence")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
        inputs["attention_mask"] = attention_mask
    if attention_mask.shape != input_ids.shape or not bool(attention_mask.bool().all()):
        raise ValueError("unpadded generation prompt must have a full attention mask")
    camera_id = layout.token_id(camera_token)
    if int(input_ids.eq(camera_id).sum()) != 1:
        raise ValueError("generation prompt must contain one matching camera token")
    for query_id in layout.role_query_ids:
        if int(input_ids.eq(query_id).sum()) != 1:
            raise ValueError("generation prompt must contain every role query exactly once")
    return inputs


def _base_forward(
    backbone: nn.Module,
    inputs: Mapping[str, Tensor],
    *,
    cache_position: Tensor,
    position_ids: Tensor | None = None,
    past_key_values: Any | None = None,
) -> Any:
    reserved = {
        "past_key_values",
        "cache_position",
        "position_ids",
        "use_cache",
        "output_hidden_states",
        "return_dict",
    }.intersection(inputs)
    if reserved:
        raise ValueError("processor inputs override Qwen decoding controls")
    return backbone(
        **inputs,
        past_key_values=past_key_values,
        cache_position=cache_position,
        position_ids=position_ids,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )


def _consume_token(
    *,
    planner: nn.Module,
    backbone: nn.Module,
    token_id: int,
    past_key_values: Any,
    rope_deltas: Tensor,
    attention_mask: Tensor,
    device: torch.device,
) -> tuple[Tensor, Any, Tensor]:
    if type(token_id) is not int or token_id < 0:
        raise RuntimeError("decoder attempted to consume an illegal token ID")
    next_attention = torch.cat(
        (
            attention_mask,
            torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ),
        dim=1,
    )
    cache_position = torch.tensor(
        [attention_mask.shape[1]],
        dtype=torch.long,
        device=device,
    )
    position_ids = cache_position.view(1, 1, 1).expand(3, 1, 1)
    position_ids = position_ids + rope_deltas.reshape(1, 1, 1)
    output = _base_forward(
        backbone,
        {
            "input_ids": torch.tensor([[token_id]], dtype=torch.long, device=device),
            "attention_mask": next_attention,
        },
        cache_position=cache_position,
        position_ids=position_ids,
        past_key_values=past_key_values,
    )
    hidden = _last_hidden(planner, output)
    if hidden.shape[0] != 1 or hidden.shape[1] != 1:
        raise RuntimeError("incremental Qwen pass must return exactly one token state")
    return hidden[:, -1], _past_key_values(output), next_attention


def _decode_camera_row(
    planner: nn.Module,
    *,
    backbone: nn.Module,
    visual_weight: Tensor,
    image: Tensor,
    instruction: str,
    camera: str,
    layout: VisualVocabularyLayout,
    processor: Any,
) -> tuple[Tensor, Tensor, Tensor]:
    device = visual_weight.device
    inputs = _build_prompt_inputs(
        processor=processor,
        layout=layout,
        image=image,
        instruction=instruction,
        camera=camera,
        device=device,
    )
    prompt_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    output = _base_forward(
        backbone,
        inputs,
        cache_position=torch.arange(
            prompt_ids.shape[1],
            dtype=torch.long,
            device=device,
        ),
    )
    prompt_hidden = _last_hidden(planner, output)
    if (
        prompt_hidden.shape[:2] != prompt_ids.shape
        or prompt_hidden.shape[-1] != visual_weight.shape[1]
    ):
        raise RuntimeError("prompt hidden states do not align with Qwen inputs")
    past = _past_key_values(output)
    rope_deltas = _rope_deltas(output, backbone=backbone, device=device)
    query_positions = torch.tensor(
        [
            int(torch.nonzero(prompt_ids[0].eq(query_id), as_tuple=False).item())
            for query_id in layout.role_query_ids
        ],
        dtype=torch.long,
        device=device,
    )
    field_hidden = prompt_hidden[:, query_positions]

    current_hidden, past, attention_mask = _consume_token(
        planner=planner,
        backbone=backbone,
        token_id=layout.token_id(PLAN_START_TOKEN),
        past_key_values=past,
        rope_deltas=rope_deltas,
        attention_mask=attention_mask,
        device=device,
    )
    frame_codes: list[Tensor] = []
    frame_post_hidden: list[Tensor] = []
    for frame_index in range(_NUM_KEYFRAMES):
        current_hidden, past, attention_mask = _consume_token(
            planner=planner,
            backbone=backbone,
            token_id=layout.token_id(FRAME_START_TOKENS[frame_index]),
            past_key_values=past,
            rope_deltas=rope_deltas,
            attention_mask=attention_mask,
            device=device,
        )
        codes: list[int] = []
        post_states: list[Tensor] = []
        for _ in range(_TOKENS_PER_FRAME):
            logits = F.linear(current_hidden, visual_weight)
            if logits.shape != (1, _VISUAL_VOCAB_SIZE):
                raise RuntimeError(
                    "visual-only logits must cover exactly 65,536 rows"
                )
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("non-finite visual logits")
            code = int(logits.argmax(dim=-1).item())
            if not 0 <= code < _VISUAL_VOCAB_SIZE:
                raise RuntimeError("decoder selected an illegal visual ID")
            token_id = layout.code_token_id(code)
            current_hidden, past, attention_mask = _consume_token(
                planner=planner,
                backbone=backbone,
                token_id=token_id,
                past_key_values=past,
                rope_deltas=rope_deltas,
                attention_mask=attention_mask,
                device=device,
            )
            codes.append(code)
            post_states.append(current_hidden.squeeze(0))
        frame_codes.append(torch.tensor(codes, dtype=torch.long, device=device))
        frame_post_hidden.append(torch.stack(post_states))
        current_hidden, past, attention_mask = _consume_token(
            planner=planner,
            backbone=backbone,
            token_id=layout.token_id(FRAME_END_TOKENS[frame_index]),
            past_key_values=past,
            rope_deltas=rope_deltas,
            attention_mask=attention_mask,
            device=device,
        )
    _consume_token(
        planner=planner,
        backbone=backbone,
        token_id=layout.token_id(PLAN_END_TOKEN),
        past_key_values=past,
        rope_deltas=rope_deltas,
        attention_mask=attention_mask,
        device=device,
    )
    return (
        torch.stack(frame_codes),
        torch.stack(frame_post_hidden),
        field_hidden.squeeze(0),
    )


@torch.no_grad()
def generate_grounded_plan(
    planner: nn.Module,
    *,
    current_images: Tensor,
    instructions: Sequence[str],
    camera_names: Sequence[str],
    layout: VisualVocabularyLayout | None = None,
    processor: Any | None = None,
) -> GeneratedGroundedPlan:
    """Greedily generate exact four-frame plans through independent KV caches."""

    layout, processor = _resolve_components(
        planner,
        layout=layout,
        processor=processor,
    )
    if (
        not isinstance(current_images, Tensor)
        or current_images.ndim != 4
        or current_images.shape[1] != 3
    ):
        raise ValueError("current_images must have shape [N,3,H,W]")
    camera_batch = int(current_images.shape[0])
    if camera_batch <= 0:
        raise ValueError("current_images must contain at least one camera row")
    if len(instructions) != camera_batch or len(camera_names) != camera_batch:
        raise ValueError("images, instructions, and camera_names must align")
    if any(type(value) is not str or not value for value in instructions):
        raise ValueError("instructions must contain nonempty strings")
    if any(camera not in CAMERA_NAMES for camera in camera_names):
        raise ValueError(f"camera_names must contain only {CAMERA_NAMES!r}")

    visual_weight = _current_visual_embedding_rows(planner, layout)
    hidden_dim = int(visual_weight.shape[1])
    if hidden_dim <= 0 or int(getattr(planner, "hidden_dim", hidden_dim)) != hidden_dim:
        raise ValueError("planner hidden width differs from visual embedding rows")
    codebook = getattr(planner, "codebook", None)
    if (
        not isinstance(codebook, Tensor)
        or codebook.ndim != 2
        or codebook.shape[0] != _VISUAL_VOCAB_SIZE
        or codebook.device != visual_weight.device
    ):
        raise ValueError("planner codebook must align with all visual rows and device")
    if bool(getattr(planner, "_enforce_released_geometry", False)):
        released_widths = (
            hidden_dim,
            int(codebook.shape[1]),
            int(getattr(planner, "text_dim", -1)),
        )
        if released_widths != (
            _GEOMETRY.qwen_hidden_dim,
            _GEOMETRY.ta_code_dim,
            _GEOMETRY.text_align_dim,
        ):
            raise ValueError(
                "released generation requires hidden/code/text widths "
                "2048/1536/1152"
            )
    for name in (
        "semantic_projection",
        "phrase_projection",
        "grounding_query",
        "fusion_gate",
        "target_times",
    ):
        if not hasattr(planner, name):
            raise TypeError(f"planner is missing the Task 7 {name} interface")

    backbone = _language_backbone(planner)
    rows = [
        _decode_camera_row(
            planner,
            backbone=backbone,
            visual_weight=visual_weight,
            image=image,
            instruction=instruction,
            camera=camera,
            layout=layout,
            processor=processor,
        )
        for image, instruction, camera in zip(
            current_images,
            instructions,
            camera_names,
        )
    ]
    codes = torch.stack([row[0] for row in rows])
    post_hidden = torch.stack([row[1] for row in rows])
    field_hidden = torch.stack([row[2] for row in rows])
    predicted_phrases = F.normalize(
        planner.phrase_projection(field_hidden),
        dim=-1,
        eps=1e-12,
    )
    semantic_features = F.normalize(
        planner.semantic_projection(post_hidden),
        dim=-1,
        eps=1e-12,
    )
    grounding_queries = F.normalize(
        planner.grounding_query(post_hidden),
        dim=-1,
        eps=1e-12,
    )
    relevance_logits = torch.einsum(
        "nktd,nrd->nkrt",
        grounding_queries,
        predicted_phrases,
    )
    relevance = relevance_logits.softmax(dim=-1)
    fusion_gate = planner.fusion_gate(
        torch.cat(
            (post_hidden, relevance_logits.permute(0, 1, 3, 2)),
            dim=-1,
        )
    )
    code_embeddings = F.embedding(codes, codebook)
    times = planner.target_times.to(post_hidden)
    values = {
        "codes": codes,
        "code_embeddings": code_embeddings,
        "post_hidden": post_hidden,
        "predicted_phrase_embeddings": predicted_phrases,
        "semantic_features": semantic_features,
        "relevance": relevance,
        "fusion_gate": fusion_gate,
        "times": times,
    }
    if getattr(planner, "_enforce_released_geometry", True) is False:
        return GeneratedGroundedPlan._from_test_components(**values)
    return GeneratedGroundedPlan(
        **values,
    )
