from __future__ import annotations

import sys

import pytest


def test_importing_qwen35_planx_does_not_import_the_legacy_planner_tree() -> None:
    legacy_modules_before = {
        name
        for name in sys.modules
        if name == "qwen3_vl_semantic_planner"
        or name.startswith("qwen3_vl_semantic_planner.")
    }

    import qwen35_planx  # noqa: F401

    legacy_modules_after = {
        name
        for name in sys.modules
        if name == "qwen3_vl_semantic_planner"
        or name.startswith("qwen3_vl_semantic_planner.")
    }
    assert legacy_modules_after == legacy_modules_before


def test_qwen35_metadata_rejects_a_legacy_qwen3vl_backend() -> None:
    from qwen35_planx.config import PlannerMetadata

    payload = PlannerMetadata.example().to_dict()
    payload["planner_backend"] = "qwen3vl2b"

    with pytest.raises(ValueError, match="planner_backend"):
        PlannerMetadata.from_dict(payload)
