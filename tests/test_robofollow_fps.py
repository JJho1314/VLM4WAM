import ast
from pathlib import Path
from typing import Any, Dict
import pytest


def test_fractional_video_stride_preserves_action_horizon():
    path=Path(__file__).resolve().parents[1]/'ge_act/runner/ge_trainer.py'
    tree=ast.parse(path.read_text())
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_effective_video_fps')
    ns={'Any':Any,'Dict':Dict}
    exec(compile(ast.Module(body=[helper],type_ignores=[]),str(path),'exec'),ns)
    fps=ns['compute_effective_video_fps']
    assert fps(dict(chunk=9,action_chunk=32,source_fps=250/15))==pytest.approx(4.6875)
    assert fps(dict(chunk=9,action_chunk=54,source_fps=250/15))==pytest.approx(250/90)
    for field in ('chunk','action_chunk','source_fps'):
        args=dict(chunk=9,action_chunk=32,source_fps=250/15);args[field]=0
        with pytest.raises(ValueError):fps(args)
