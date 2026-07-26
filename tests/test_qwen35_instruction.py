from __future__ import annotations


def test_parse_pick_place_instruction() -> None:
    from qwen35_planx.instruction import parse_libero_instruction

    fields = parse_libero_instruction(
        "pick up the black bowl between the plate and the ramekin"
    )

    assert fields.action == "pick up and place"
    assert fields.source == "the black bowl"
    assert fields.target == "between the plate and the ramekin"
    assert fields.confidences == (1.0, 1.0, 1.0)


def test_parse_pick_place_instruction_with_object_pronoun() -> None:
    from qwen35_planx.instruction import parse_libero_instruction

    fields = parse_libero_instruction("pick up the black bowl and place it on the plate")

    assert (fields.action, fields.source, fields.target) == (
        "pick up and place",
        "the black bowl",
        "on the plate",
    )


def test_parse_drawer_and_stove_instructions_with_missing_target() -> None:
    from qwen35_planx.instruction import parse_libero_instruction

    drawer = parse_libero_instruction("Open the bottom drawer.")
    stove = parse_libero_instruction("turn on the stove")

    assert (drawer.action, drawer.source, drawer.target) == (
        "open",
        "the bottom drawer",
        "",
    )
    assert drawer.confidences == (1.0, 0.0, 1.0)
    assert (stove.action, stove.source, stove.target) == ("turn on", "the stove", "")
    assert stove.confidences == (1.0, 0.0, 1.0)


def test_parse_basket_and_spatial_relation_instructions() -> None:
    from qwen35_planx.instruction import parse_libero_instruction

    basket = parse_libero_instruction("put the white bowl in the basket")
    spatial = parse_libero_instruction("place the mug next to the plate")

    assert (basket.action, basket.source, basket.target) == (
        "put",
        "the white bowl",
        "in the basket",
    )
    assert (spatial.action, spatial.source, spatial.target) == (
        "place",
        "the mug",
        "next to the plate",
    )


def test_parse_unrecognized_instruction_preserves_original_and_marks_missing_fields() -> None:
    from qwen35_planx.instruction import parse_libero_instruction

    instruction = "  Carefully   rearrange, everything!  "
    fields = parse_libero_instruction(instruction)

    assert fields.original == instruction
    assert (fields.action, fields.source, fields.target) == ("", "", "")
    assert fields.confidences == (0.0, 0.0, 0.0)


def test_format_grounded_prompt_is_idempotent() -> None:
    from qwen35_planx.instruction import format_grounded_prompt, parse_libero_instruction

    fields = parse_libero_instruction("put the black bowl on the plate")

    expected = (
        "<ACT>put</ACT>\n"
        "<SRC>the black bowl</SRC>\n"
        "<TGT>on the plate</TGT>\n"
        "Instruction: put the black bowl on the plate\n"
        "<SRC_QUERY><TGT_QUERY><ACT_QUERY>\n"
        "Predict four future semantic frames."
    )
    assert format_grounded_prompt(fields) == expected
    assert format_grounded_prompt(fields) == expected


def test_counterfactual_changes_exactly_one_field() -> None:
    from qwen35_planx.instruction import (
        InstructionVocabulary,
        build_counterfactuals,
        parse_libero_instruction,
    )

    fields = parse_libero_instruction("put the black bowl on the plate")
    vocab = InstructionVocabulary(
        actions=("put", "open"),
        sources=("the black bowl", "the white bowl"),
        targets=("on the plate", "in the basket"),
    )
    negatives = build_counterfactuals(fields, vocab, max_per_field=1)

    assert [item.changed_field for item in negatives] == [
        "action",
        "source",
        "target",
    ]
    assert all(item.fields != fields for item in negatives)
    assert negatives[0].fields.source == fields.source
    assert negatives[0].fields.target == fields.target
    assert negatives[1].fields.action == fields.action
    assert negatives[1].fields.target == fields.target
    assert negatives[2].fields.action == fields.action
    assert negatives[2].fields.source == fields.source


def test_vocabulary_normalizes_to_a_deterministic_order() -> None:
    from qwen35_planx.instruction import InstructionVocabulary

    vocabulary = InstructionVocabulary(
        actions=("put", "open", "put"),
        sources=("the white bowl", "the black bowl"),
        targets=("on the plate", "in the basket"),
    )

    assert vocabulary.actions == ("open", "put")
    assert vocabulary.sources == ("the black bowl", "the white bowl")
    assert vocabulary.targets == ("in the basket", "on the plate")
