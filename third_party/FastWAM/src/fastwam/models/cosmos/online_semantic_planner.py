"""Import the external frozen planner without coupling FastWAM to its package."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import uuid


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

    module_path = Path(code_dir).expanduser().resolve() / "dino_depth_plan_provider.py"
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(
            f"online planner provider not found: {module_path}"
        )
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"online planner checkpoint directory not found: {checkpoint_path}"
        )
    return module_path, checkpoint_path


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
    trainer_dir = module_path.parent.parent
    path_digest = hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_fastwam_online_planner_{path_digest}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for online planner: {module_path}")
    module = importlib.util.module_from_spec(spec)

    trainer_path = str(trainer_dir)
    inserted_trainer_path = trainer_path not in sys.path
    if inserted_trainer_path:
        sys.path.insert(0, trainer_path)
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
        if inserted_trainer_path:
            try:
                sys.path.remove(trainer_path)
            except ValueError:
                pass
        # Provider methods retain their module globals directly; retaining this
        # one-off import in sys.modules only leaks stale implementations on reload.
        sys.modules.pop(module_name, None)
