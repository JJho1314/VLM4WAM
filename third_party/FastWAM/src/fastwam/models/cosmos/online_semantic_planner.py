"""Import the external frozen planner without coupling FastWAM to its package."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import uuid


_REQUIRED_CHECKPOINT_FILES = (
    "plan_head.pt",
    "depth_head.pt",
    "current_plan_head.pt",
    "current_depth_head.pt",
    "plan_token_embedding.pt",
    "planner_meta.json",
)
_REQUIRED_CHECKPOINT_DIRS = (
    "qwen3vl_lora_or_model",
    "processor",
)
_EXPECTED_METADATA = {
    "sequence_length": 9,
    "num_keyframes": 1,
    "grid_size": 16,
    "semantic_dim": 1024,
    "target_tokens": 256,
    "keyframe_offsets": [8],
    "keyframe_scheme": "even_future",
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
    "planner_input_frame": "fastwam_current_multicamera_composite",
    "token_order": "keyframe_major_row_major",
}
_EXPECTED_NORMALIZED_TIMES = (1.0,)
_REQUIRED_FINITE_NUMERIC_METADATA = (
    "mse_loss_weight",
    "cosine_loss_weight",
    "norm_loss_weight",
    "variance_loss_weight",
    "infonce_loss_weight",
    "infonce_temperature",
    "depth_loss_weight",
    "current_dino_loss_weight",
    "future_dino_loss_weight",
    "current_depth_loss_weight",
    "future_depth_loss_weight",
)
_PROVIDER_FILENAME = "dino_depth_plan_provider.py"


def _is_finite_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _metadata_value_matches(actual, expected) -> bool:
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return _is_finite_number(actual) and float(actual) == expected
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _metadata_value_matches(item, expected_item)
                for item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def _validate_checkpoint_export(checkpoint_path: Path) -> None:
    missing = [
        name
        for name in _REQUIRED_CHECKPOINT_FILES
        if not (checkpoint_path / name).is_file()
    ]
    missing.extend(
        name
        for name in _REQUIRED_CHECKPOINT_DIRS
        if not (checkpoint_path / name).is_dir()
    )
    if missing:
        raise FileNotFoundError(
            f"incomplete online planner checkpoint {checkpoint_path}: "
            f"missing {missing}"
        )

    metadata_path = checkpoint_path / "planner_meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid online planner metadata {metadata_path}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError(
            f"invalid online planner metadata {metadata_path}: expected an object"
        )
    independent = metadata.get("independent_modality_task_tokens", False)
    if type(independent) is not bool:
        raise ValueError(
            "incompatible online planner metadata field "
            "independent_modality_task_tokens: expected a boolean, "
            f"got {independent!r}"
        )
    num_task_tokens = metadata.get("num_task_tokens", 8)
    if type(num_task_tokens) is not int or num_task_tokens <= 0:
        raise ValueError(
            "incompatible online planner metadata field num_task_tokens: "
            f"expected a positive integer, got {num_task_tokens!r}"
        )
    expected_metadata = dict(_EXPECTED_METADATA)
    expected_metadata.update(
        {
            "num_task_tokens": num_task_tokens,
            "branch_latent_per_keyframe": num_task_tokens,
            "total_unique_latent_per_keyframe": 2 * num_task_tokens,
            "latent_len": 2 * num_task_tokens,
            "query_layout": (
                f"current_{num_task_tokens}_then_future_{num_task_tokens}__"
                "dino_depth_shared_within_time"
            ),
        }
    )
    if independent:
        expected_metadata.update(
            {
                "total_unique_latent_per_keyframe": 4 * num_task_tokens,
                "latent_len": 4 * num_task_tokens,
                "query_layout": (
                    f"current_dino_{num_task_tokens}_then_"
                    f"current_depth_{num_task_tokens}_then_"
                    f"future_dino_{num_task_tokens}_then_"
                    f"future_depth_{num_task_tokens}"
                ),
            }
        )
    for field, expected in expected_metadata.items():
        actual = metadata.get(field)
        if not _metadata_value_matches(actual, expected):
            raise ValueError(
                f"incompatible online planner metadata field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )

    for field in _REQUIRED_FINITE_NUMERIC_METADATA:
        actual = metadata.get(field)
        if not _is_finite_number(actual):
            raise ValueError(
                f"incompatible online planner metadata field {field}: "
                f"expected a finite number, got {actual!r}"
            )

    plan_token_ids = metadata.get("plan_token_ids")
    if (
        not isinstance(plan_token_ids, list)
        or len(plan_token_ids) != expected_metadata["latent_len"]
        or any(
            type(token_id) is not int or token_id < 0
            for token_id in plan_token_ids
        )
        or len(set(plan_token_ids)) != len(plan_token_ids)
    ):
        raise ValueError(
            "incompatible online planner metadata field plan_token_ids: "
            f"expected {expected_metadata['latent_len']} unique non-negative integers"
        )

    raw_plan_tokens = metadata.get("plan_token_strings", ())
    try:
        actual_plan_tokens = tuple(raw_plan_tokens)
    except TypeError as error:
        raise ValueError(
            "incompatible online planner metadata field plan_token_strings: "
            "expected the exact exported query tokens"
        ) from error
    expected_plan_token_strings = tuple(
        f"<|sem_plan_{index}|>"
        for index in range(expected_metadata["latent_len"])
    )
    if actual_plan_tokens != expected_plan_token_strings:
        raise ValueError(
            "incompatible online planner metadata field plan_token_strings: "
            "expected the exact exported query tokens"
        )

    raw_times = metadata.get("normalized_keyframe_times")
    if (
        not isinstance(raw_times, list)
        or len(raw_times) != len(_EXPECTED_NORMALIZED_TIMES)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_times
        )
        or any(
            abs(float(actual) - expected) > 1e-7
            for actual, expected in zip(
                raw_times,
                _EXPECTED_NORMALIZED_TIMES,
                strict=True,
            )
        )
    ):
        raise ValueError(
            "incompatible online planner metadata field "
            "normalized_keyframe_times: expected "
            f"{list(_EXPECTED_NORMALIZED_TIMES)!r}, got {raw_times!r}"
        )


def _online_planner_code_candidates(code_dir: str | Path) -> tuple[Path, ...]:
    """Return deterministic code-root candidates for a relative config path."""
    requested = Path(code_dir).expanduser()
    if requested.is_absolute():
        return (requested.resolve(),)

    roots = []
    explicit_root = os.environ.get("VLM4WAM_ROOT")
    if explicit_root and explicit_root.strip():
        roots.append(Path(explicit_root).expanduser())
    roots.append(Path.cwd())
    roots.extend(Path(__file__).resolve().parents)

    candidates = []
    seen = set()
    for root in roots:
        candidate = (root / requested).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return tuple(candidates)


def validate_online_semantic_planner_paths(
    *,
    code_dir: str,
    checkpoint_dir: str,
) -> tuple[Path, Path]:
    """Resolve the two online-planner roots before any heavyweight allocation."""
    if not isinstance(code_dir, (str, Path)) or not str(code_dir).strip():
        raise ValueError("online planner code_dir must be a non-empty path")
    if not isinstance(checkpoint_dir, (str, Path)) or not str(checkpoint_dir).strip():
        raise ValueError("online planner checkpoint_dir must be a non-empty path")

    code_candidates = _online_planner_code_candidates(code_dir)
    module_candidates = tuple(
        candidate / _PROVIDER_FILENAME for candidate in code_candidates
    )
    module_path = next(
        (candidate for candidate in module_candidates if candidate.is_file()),
        None,
    )
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    if module_path is None:
        searched = "\n".join(f"  - {candidate}" for candidate in module_candidates)
        raise FileNotFoundError(
            "online planner provider not found; searched:\n"
            f"{searched}\n"
            "Set VLM4WAM_ROOT or pass an absolute online planner code_dir."
        )
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"online planner checkpoint directory not found: {checkpoint_path}"
        )
    _validate_checkpoint_export(checkpoint_path)
    return module_path, checkpoint_path


def _local_top_level_import_names(*roots: Path) -> set[str]:
    names = set()
    for root in roots:
        try:
            entries = tuple(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_file() and entry.suffix == ".py" and entry.stem != "__init__":
                names.add(entry.stem)
            elif entry.is_dir() and (entry / "__init__.py").is_file():
                names.add(entry.name)
    return names


def _module_is_under_roots(module, roots: tuple[Path, ...]) -> bool:
    raw_paths = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        raw_paths.append(module_file)
    module_search_path = getattr(module, "__path__", None)
    if module_search_path is not None:
        try:
            raw_paths.extend(module_search_path)
        except TypeError:
            pass
    for raw_path in raw_paths:
        try:
            resolved = Path(raw_path).resolve()
        except (OSError, TypeError, ValueError):
            continue
        if any(resolved == root or resolved.is_relative_to(root) for root in roots):
            return True
    return False


def load_online_semantic_planner(
    *,
    code_dir: str,
    checkpoint_dir: str,
    device,
    dtype,
):
    """Load one frozen provider instance from a validated local export.

    A unique module name avoids serving a stale provider when multiple planner
    checkouts are exercised in one process.  The trainer directory is exposed
    only for the duration of provider construction because the provider lazily
    imports the adjacent trainer helpers in ``from_checkpoint``.
    """
    module_path, checkpoint_path = validate_online_semantic_planner_paths(
        code_dir=code_dir,
        checkpoint_dir=checkpoint_dir,
    )
    code_path = module_path.parent.resolve()
    trainer_dir = code_path.parent.resolve()
    path_digest = hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_fastwam_online_planner_{path_digest}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for online planner: {module_path}")
    module = importlib.util.module_from_spec(spec)

    original_sys_path = list(sys.path)
    original_modules = dict(sys.modules)
    local_roots = (trainer_dir, code_path)
    local_names = _local_top_level_import_names(*local_roots)
    for loaded_name in tuple(sys.modules):
        if any(
            loaded_name == local_name
            or loaded_name.startswith(f"{local_name}.")
            for local_name in local_names
        ):
            sys.modules.pop(loaded_name, None)
    importlib.invalidate_caches()
    sys.path[0:0] = [str(code_path), str(trainer_dir)]
    sys.modules[module_name] = module
    try:
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise ImportError(
                f"failed to import online planner provider from {module_path}: {error}"
            ) from error

        provider_class = getattr(module, "FrozenDinoDepthPlanProvider", None)
        factory = getattr(provider_class, "from_checkpoint", None)
        if not callable(factory):
            raise ImportError(
                f"online planner provider {module_path} must define "
                "FrozenDinoDepthPlanProvider.from_checkpoint"
            )
        return factory(
            checkpoint_path,
            device=device,
            dtype=dtype,
        )
    finally:
        sys.path[:] = original_sys_path
        # Restore every pre-existing module mapping that provider code replaced
        # or removed. Newly imported third-party dependencies remain cached.
        for name, original_module in original_modules.items():
            if sys.modules.get(name) is not original_module:
                sys.modules[name] = original_module
        # Remove only newly imported modules physically owned by this external
        # checkout. This prevents canonical sibling names leaking across checkouts.
        for name, imported_module in tuple(sys.modules.items()):
            if name in original_modules:
                continue
            if _module_is_under_roots(imported_module, local_roots):
                sys.modules.pop(name, None)
