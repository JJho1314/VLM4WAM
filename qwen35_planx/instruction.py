"""Structured, deterministic LIBERO instruction interfaces for grounding."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable

from qwen35_planx.config import _GROUNDING_ROLES

_RELATION = (
    r"(?:on|onto|in|into|inside|next to|between|behind|in front of|"
    r"to the left of|to the right of|left of|right of|near|under|above)"
)
_PICK_AND_PLACE = re.compile(
    rf"^pick up (?P<source>.+?) (?:and )?(?:place|put) (?:it )?(?P<target>{_RELATION} .+)$"
)
_PICK_WITH_RELATION = re.compile(
    rf"^pick up (?P<source>.+?) (?P<target>{_RELATION} .+)$"
)
_PLACE_WITH_RELATION = re.compile(
    rf"^(?P<action>put|place|move|stack) (?P<source>.+?) (?P<target>{_RELATION} .+)$"
)
_OBJECT_ACTION = re.compile(
    r"^(?P<action>turn on|turn off|open|close) (?P<source>.+)$"
)
_PARTIAL_PLACE = re.compile(r"^(?P<action>put|place|move|stack|pick up) (?P<source>.+)$")


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().strip(".,!?;:").lower()


@dataclass(frozen=True)
class InstructionFields:
    """The original instruction and source/target/action grounding phrases."""

    original: str
    action: str
    source: str
    target: str
    confidences: tuple[float, float, float]

    def __post_init__(self) -> None:
        if len(self.confidences) != len(_GROUNDING_ROLES):
            raise ValueError("confidences must contain source, target, and action")
        if any(not 0.0 <= confidence <= 1.0 for confidence in self.confidences):
            raise ValueError("confidences must be between zero and one")


@dataclass(frozen=True)
class CounterfactualInstruction:
    """One controlled source, target, or action substitution."""

    changed_field: str
    fields: InstructionFields

    def __post_init__(self) -> None:
        if self.changed_field not in _GROUNDING_ROLES:
            raise ValueError(f"unknown grounding field: {self.changed_field!r}")


@dataclass(frozen=True)
class InstructionVocabulary:
    """Canonical sorted substitutions used to form deterministic negatives."""

    actions: tuple[str, ...]
    sources: tuple[str, ...]
    targets: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("actions", "sources", "targets"):
            values = tuple(sorted({_normalise(value) for value in getattr(self, name)}))
            if any(not value for value in values):
                raise ValueError(f"{name} must not contain empty values")
            object.__setattr__(self, name, values)


def _fields(
    original: str, action: str = "", source: str = "", target: str = ""
) -> InstructionFields:
    action, source, target = (_normalise(value) for value in (action, source, target))
    confidences = (
        1.0 if source else 0.0,
        1.0 if target else 0.0,
        1.0 if action else 0.0,
    )
    return InstructionFields(original, action, source, target, confidences)


def parse_libero_instruction(instruction: str) -> InstructionFields:
    """Parse ordered LIBERO templates while retaining the supplied instruction."""

    normalized = _normalise(instruction)
    for template in (_PICK_AND_PLACE, _PICK_WITH_RELATION):
        match = template.fullmatch(normalized)
        if match:
            return _fields(
                instruction,
                action="pick up and place",
                source=match.group("source"),
                target=match.group("target"),
            )
    match = _PLACE_WITH_RELATION.fullmatch(normalized)
    if match:
        return _fields(
            instruction,
            action=match.group("action"),
            source=match.group("source"),
            target=match.group("target"),
        )
    match = _OBJECT_ACTION.fullmatch(normalized)
    if match:
        return _fields(instruction, action=match.group("action"), source=match.group("source"))
    match = _PARTIAL_PLACE.fullmatch(normalized)
    if match:
        action = match.group("action")
        return _fields(
            instruction,
            action="pick up and place" if action == "pick up" else action,
            source=match.group("source"),
        )
    return _fields(instruction)


def format_grounded_prompt(fields: InstructionFields) -> str:
    """Return the fixed textual marker order consumed by the grounded planner."""

    return (
        f"<ACT>{fields.action}</ACT>\n"
        f"<SRC>{fields.source}</SRC>\n"
        f"<TGT>{fields.target}</TGT>\n"
        f"Instruction: {fields.original}\n"
        "<SRC_QUERY><TGT_QUERY><ACT_QUERY>\n"
        "Predict four future semantic frames."
    )


def _substitutions(values: Iterable[str], current: str) -> tuple[str, ...]:
    return tuple(value for value in values if value != current)


def build_counterfactuals(
    fields: InstructionFields,
    vocabulary: InstructionVocabulary,
    *,
    max_per_field: int = 1,
) -> list[CounterfactualInstruction]:
    """Build controlled negatives, changing one canonical grounding role at a time."""

    if max_per_field < 0:
        raise ValueError("max_per_field must be non-negative")
    candidates = {
        "action": _substitutions(vocabulary.actions, fields.action),
        "source": _substitutions(vocabulary.sources, fields.source),
        "target": _substitutions(vocabulary.targets, fields.target),
    }
    negatives: list[CounterfactualInstruction] = []
    for field in ("action", "source", "target"):
        for value in candidates[field][:max_per_field]:
            confidences = list(fields.confidences)
            confidences[_GROUNDING_ROLES.index(field)] = 1.0
            negatives.append(
                CounterfactualInstruction(
                    changed_field=field,
                    fields=replace(fields, **{field: value}, confidences=tuple(confidences)),
                )
            )
    return negatives
