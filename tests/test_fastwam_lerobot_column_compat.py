from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
COMPAT_PATH = (
    ROOT / "third_party/FastWAM/src/fastwam/datasets/lerobot/lerobot/column_compat.py"
)
LEROBOT_DATASET_PATH = (
    ROOT / "third_party/FastWAM/src/fastwam/datasets/lerobot/lerobot/lerobot_dataset.py"
)


class ColumnLike:
    """Minimal iterable with the behavior relevant to datasets.Column."""

    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def load_compat_module():
    assert COMPAT_PATH.is_file(), "FastWAM Column compatibility helper is missing"
    spec = importlib.util.spec_from_file_location(
        "fastwam_lerobot_column_compat",
        COMPAT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "values",
    [
        [torch.tensor(1, dtype=torch.int64), torch.tensor(2, dtype=torch.int64)],
        [
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([3.0, 4.0], dtype=torch.float32),
        ],
    ],
)
def test_stack_hf_column_materializes_column_like_iterables(values):
    column = ColumnLike(values)
    with pytest.raises(TypeError, match="must be tuple of Tensors"):
        torch.stack(column)

    compat = load_compat_module()
    result = compat.stack_hf_column(column)

    assert torch.equal(result, torch.stack(tuple(values)))
    assert result.dtype == values[0].dtype


def test_stack_hf_column_returns_tensor_inputs_unchanged():
    compat = load_compat_module()
    tensor = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    result = compat.stack_hf_column(tensor)

    assert result is tensor


def test_lerobot_dataset_routes_every_hf_column_stack_through_helper():
    source = LEROBOT_DATASET_PATH.read_text(encoding="utf-8")

    assert "from .column_compat import stack_hf_column" in source
    assert "torch.stack(self.hf_dataset" not in source
    assert "torch.stack(timestamps)" not in source
    assert "torch.stack(selected_data[key])" not in source
    assert source.count("stack_hf_column(") == 6
