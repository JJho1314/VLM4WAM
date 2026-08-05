"""Stable SHA-256 helpers for Baton artifact contracts."""

from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Any


def sha256_json(value: Any) -> str:
    """Return the SHA-256 of canonical JSON-compatible data."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Stream *path* into SHA-256 without loading the full artifact."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_artifact(path: str | PathLike[str]) -> str:
    """Hash a file or an immutable directory tree including relative paths."""

    artifact = Path(path)
    if artifact.is_file():
        return sha256_file(artifact)
    if not artifact.is_dir():
        raise FileNotFoundError(artifact)
    entries = [
        (
            child.relative_to(artifact).as_posix(),
            child.stat().st_size,
            sha256_file(child),
        )
        for child in sorted(artifact.rglob("*"))
        if child.is_file()
    ]
    if not entries:
        raise ValueError(f"artifact directory is empty: {artifact}")
    return sha256_json(entries)
