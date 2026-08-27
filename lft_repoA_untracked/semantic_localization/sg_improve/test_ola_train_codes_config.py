import ast
from pathlib import Path


SOURCE = Path(__file__).with_name("ola_train_codes.py")


def _tree():
    return ast.parse(SOURCE.read_text())


def test_gradient_checkpointing_is_an_explicit_environment_setting():
    assignments = {
        target.id: node.value
        for node in _tree().body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    assert "GRADIENT_CHECKPOINTING" in assignments
    assert "GRADIENT_CHECKPOINTING" in ast.unparse(assignments["GRADIENT_CHECKPOINTING"])


def test_planner_does_not_unconditionally_enable_checkpointing():
    planner = next(
        node
        for node in _tree().body
        if isinstance(node, ast.ClassDef) and node.name == "Planner"
    )
    init = next(
        node
        for node in planner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "__init__"
    )
    args = [arg.arg for arg in init.args.args]
    assert "gradient_checkpointing" in args

    enable_calls = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "gradient_checkpointing_enable"
    ]
    assert len(enable_calls) == 1
    assert any(
        isinstance(parent, ast.If)
        and "gradient_checkpointing" in ast.unparse(parent.test)
        and enable_calls[0] in list(ast.walk(parent))
        for parent in ast.walk(init)
    )


def test_checkpoint_metadata_records_batch_and_checkpointing_mode():
    source = SOURCE.read_text()
    assert '"batch_size": BS' in source
    assert '"gradient_checkpointing": GRADIENT_CHECKPOINTING' in source
