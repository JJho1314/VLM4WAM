### Task 1: Shared `libero_tgt_v1` Preprocessor

**Files:**
- Create: `qwen3_vl_semantic_planner/libero_target_text.py`
- Create: `tests/test_libero_target_text.py`

**Interfaces:**
- Produces: `TGT_MARKER: str`, `LIBERO_TGT_PREPROCESSING: str`, `InstructionPreprocessingError`, `mark_libero_target(instruction: str) -> str`, `validate_instruction_preprocessing(preprocessing: str | None) -> str | None`, and `preprocess_libero_instructions(instructions: Sequence[str], *, preprocessing: str | None) -> list[str]`.
- Consumes: no project model, dataset, PyTorch, or Transformers objects.

- [ ] **Step 1: Write failing transformation and validation tests**

```python
# tests/test_libero_target_text.py
from __future__ import annotations

import pytest

from qwen3_vl_semantic_planner.libero_target_text import (
    InstructionPreprocessingError,
    LIBERO_TGT_PREPROCESSING,
    mark_libero_target,
    preprocess_libero_instructions,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "pick up the black bowl between the plate and the ramekin",
            "pick up the [TGT] black bowl between the plate and the ramekin",
        ),
        (
            "open the middle drawer of the cabinet",
            "open the [TGT] middle drawer of the cabinet",
        ),
        (
            "turn on the stove and put the moka pot on it",
            "turn on the [TGT] stove and put the moka pot on it",
        ),
        ("put the bowl on the plate", "put the [TGT] bowl on the plate"),
        (
            "pick up the alphabet soup and place it in the basket",
            "pick up the [TGT] alphabet soup and place it in the basket",
        ),
    ],
)
def test_mark_libero_target_uses_first_direct_object(
    raw: str, expected: str
) -> None:
    assert mark_libero_target(raw) == expected


def test_mark_libero_target_is_idempotent() -> None:
    marked = "open the [TGT] top drawer and put the bowl inside"
    assert mark_libero_target(marked) == marked


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "non-empty"),
        ("wait for the robot", "no target object"),
        ("pick up the [TGT] bowl and place the [TGT] bowl", "exactly one"),
    ],
)
def test_mark_libero_target_rejects_invalid_text(
    value: str, message: str
) -> None:
    with pytest.raises(InstructionPreprocessingError, match=message):
        mark_libero_target(value)


def test_mark_libero_target_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        mark_libero_target(7)  # type: ignore[arg-type]


def test_batch_preprocessing_selects_target_or_legacy_contract() -> None:
    raw = ["put the bowl on the plate"]
    assert preprocess_libero_instructions(
        raw,
        preprocessing=LIBERO_TGT_PREPROCESSING,
    ) == ["put the [TGT] bowl on the plate"]
    assert preprocess_libero_instructions(
        raw,
        preprocessing=None,
    ) == raw
    with pytest.raises(ValueError, match="unsupported instruction preprocessing"):
        preprocess_libero_instructions(raw, preprocessing="libero_tgt_v2")
```

- [ ] **Step 2: Run the focused test and confirm the missing-module failure**

Run:

```bash
cd /data/user/jhe724/workspace/VLM4WAM_joint_geact_02b89af
pytest -q tests/test_libero_target_text.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'qwen3_vl_semantic_planner.libero_target_text'`.

- [ ] **Step 3: Implement the dependency-free preprocessor**

```python
# qwen3_vl_semantic_planner/libero_target_text.py
"""Versioned, fail-closed target marking for LIBERO instructions."""

from __future__ import annotations

import re
from collections.abc import Sequence


TGT_MARKER = "[TGT]"
LIBERO_TGT_PREPROCESSING = "libero_tgt_v1"

_VERB_PATTERN = re.compile(
    r"\b("
    r"turn on|turn off|switch on|switch off|pick up|"
    r"put|place|open|close|push|pull|pick|grab|grasp|take|get|"
    r"move|bring|shift|lift|remove|slide|pour|hang|insert|stack|"
    r"cover|uncover|rotate|press|wipe|fold|unfold|transfer|"
    r"align|arrange|position|release|drop|scoop|roll|connect|"
    r"attach|detach|hold|carry|collect"
    r")\s+",
    flags=re.IGNORECASE,
)
_STOP_PATTERN = re.compile(
    r"(?="
    r"\s+(?:in|into|inside|on|onto|to|from|out of|off|over|under|"
    r"closer|away|toward|towards|near|next to|with|using|through|"
    r"back|around|behind|beside)\b"
    r"|\s*,"
    r"|\s+and\s+(?:then\s+)?(?:put|place|move|pick|pick up|take|"
    r"open|close|flip|turn|turn on|turn off|slide|hang|pour|"
    r"remove|push|pull)\b"
    r"|\s+then\s+"
    r"|[.]"
    r"|$"
    r")",
    flags=re.IGNORECASE,
)
_ARTICLE_PATTERN = re.compile(
    r"^(?P<article>(?:the|a|an|this|that|these|those|another|one)\s+)"
    r"(?P<rest>.+)$",
    flags=re.IGNORECASE,
)
_BAD_OBJECTS = frozenset({"it", "them", "this", "that", "there"})


class InstructionPreprocessingError(ValueError):
    """A LIBERO instruction cannot satisfy the selected prompt contract."""


def validate_instruction_preprocessing(
    preprocessing: str | None,
) -> str | None:
    if preprocessing not in (None, LIBERO_TGT_PREPROCESSING):
        raise ValueError(
            "unsupported instruction preprocessing "
            f"{preprocessing!r}; expected None or "
            f"{LIBERO_TGT_PREPROCESSING!r}"
        )
    return preprocessing


def _insert_marker(phrase: str) -> str:
    article = _ARTICLE_PATTERN.match(phrase)
    if article is None:
        return f"{TGT_MARKER} {phrase}"
    return (
        f"{article.group('article')}{TGT_MARKER} "
        f"{article.group('rest')}"
    )


def mark_libero_target(instruction: str) -> str:
    if not isinstance(instruction, str):
        raise TypeError(
            "LIBERO instruction must be a string, "
            f"got {type(instruction).__name__}"
        )
    text = instruction.strip()
    if not text:
        raise InstructionPreprocessingError(
            "LIBERO instruction must be non-empty"
        )
    marker_count = text.count(TGT_MARKER)
    if marker_count == 1:
        return text
    if marker_count > 1:
        raise InstructionPreprocessingError(
            "LIBERO instruction must contain exactly one [TGT] marker: "
            f"{instruction!r}"
        )

    for verb_match in _VERB_PATTERN.finditer(text):
        object_start = verb_match.end()
        stop_match = _STOP_PATTERN.search(text, object_start)
        object_end = stop_match.start() if stop_match is not None else len(text)
        phrase = text[object_start:object_end].strip(" ,.;")
        if not phrase or phrase.lower() in _BAD_OBJECTS:
            continue
        marked = text[:object_start] + _insert_marker(phrase) + text[object_end:]
        if marked.count(TGT_MARKER) != 1:
            raise InstructionPreprocessingError(
                "target rewrite did not produce exactly one [TGT] marker: "
                f"{instruction!r}"
            )
        return marked

    raise InstructionPreprocessingError(
        f"LIBERO instruction has no target object: {instruction!r}"
    )


def preprocess_libero_instructions(
    instructions: Sequence[str],
    *,
    preprocessing: str | None,
) -> list[str]:
    validate_instruction_preprocessing(preprocessing)
    if isinstance(instructions, (str, bytes)) or not isinstance(
        instructions, Sequence
    ):
        raise TypeError("instructions must be a sequence of strings")
    values = list(instructions)
    if preprocessing == LIBERO_TGT_PREPROCESSING:
        return [mark_libero_target(value) for value in values]
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(
                "LIBERO instruction must be a string, "
                f"got {type(value).__name__}"
            )
        if not value.strip():
            raise InstructionPreprocessingError(
                "LIBERO instruction must be non-empty"
            )
        normalized.append(value)
    return normalized
```

- [ ] **Step 4: Run the preprocessor tests**

Run:

```bash
pytest -q tests/test_libero_target_text.py
```

Expected: `11 passed`, with no warnings from model libraries because the module has no model dependencies.

- [ ] **Step 5: Commit the shared contract**

```bash
git add qwen3_vl_semantic_planner/libero_target_text.py \
  tests/test_libero_target_text.py
git commit -m "feat: add LIBERO target text preprocessing"
```

---

