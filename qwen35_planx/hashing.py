"""Stable SHA-256 helpers used to bind Plan-X artifacts together."""

from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Any


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using a canonical serialization."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into SHA-256 without loading it fully into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
