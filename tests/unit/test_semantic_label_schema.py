"""Contract tests for the versioned semantic label schema."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from document_intelligence.extraction import (
    EntityType,
    LabelDefinition,
    LabelPrefix,
    SemanticLabelSchema,
    load_semantic_label_schema,
    validate_bio_sequence,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
LABEL_SCHEMA_PATH = REPOSITORY_ROOT / "configs" / "extraction" / "semantic_labels_v1.json"
ANNOTATION_SCHEMA_PATH = REPOSITORY_ROOT / "datasets" / "schemas" / "entity-annotation.schema.json"
ANNOTATION_FIXTURE_PATH = (
    REPOSITORY_ROOT / "datasets" / "fixtures" / "synthetic-entity-annotation.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as json_file:
        value: dict[str, Any] = json.load(json_file)
    return value


@pytest.fixture
def label_schema() -> SemanticLabelSchema:
    return load_semantic_label_schema(LABEL_SCHEMA_PATH)


def test_label_schema_has_stable_complete_bio_mapping(
    label_schema: SemanticLabelSchema,
) -> None:
    assert label_schema.label_to_id["O"] == 0
    assert label_schema.id_to_label[10] == "I-INVOICE_DATE"
    assert len(label_schema.labels) == 1 + 2 * len(EntityType)
    for entity_type in EntityType:
        assert f"B-{entity_type.value}" in label_schema.label_to_id
        assert f"I-{entity_type.value}" in label_schema.label_to_id


def test_mapping_properties_do_not_expose_mutable_internal_state(
    label_schema: SemanticLabelSchema,
) -> None:
    mapping = label_schema.label_to_id
    mapping["O"] = 999

    assert label_schema.label_to_id["O"] == 0


@pytest.mark.parametrize(
    "labels",
    [
        ("B-SELLER_NAME", "I-SELLER_NAME", "O", "B-TOTAL_AMOUNT"),
        ("O", "B-INVOICE_NUMBER", "B-INVOICE_NUMBER"),
        (),
    ],
)
def test_bio_validator_accepts_valid_sequences(
    label_schema: SemanticLabelSchema, labels: tuple[str, ...]
) -> None:
    validate_bio_sequence(labels, schema=label_schema)


@pytest.mark.parametrize(
    "labels",
    [
        ("I-SELLER_NAME",),
        ("O", "I-INVOICE_NUMBER"),
        ("B-SELLER_NAME", "I-INVOICE_NUMBER"),
        ("UNKNOWN",),
    ],
)
def test_bio_validator_rejects_invalid_transitions_and_labels(
    label_schema: SemanticLabelSchema, labels: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match=r"invalid BIO transition|unsupported semantic label"):
        validate_bio_sequence(labels, schema=label_schema)


def test_label_schema_rejects_non_contiguous_ids(label_schema: SemanticLabelSchema) -> None:
    invalid_labels = list(label_schema.labels)
    invalid_labels[1] = LabelDefinition(
        label_id=99,
        name="B-ORDER_REFERENCE",
        prefix=LabelPrefix.BEGIN,
        entity_type=EntityType.ORDER_REFERENCE,
    )

    with pytest.raises(ValueError, match="contiguous"):
        SemanticLabelSchema("1.0.0", "BIO", tuple(invalid_labels))


def test_annotation_schema_and_synthetic_fixture_are_valid() -> None:
    annotation_schema = _load_json(ANNOTATION_SCHEMA_PATH)
    annotation = _load_json(ANNOTATION_FIXTURE_PATH)

    Draft202012Validator.check_schema(annotation_schema)
    Draft202012Validator(annotation_schema).validate(annotation)


def test_annotation_schema_labels_match_canonical_vocabulary(
    label_schema: SemanticLabelSchema,
) -> None:
    annotation_schema = _load_json(ANNOTATION_SCHEMA_PATH)
    annotation_labels = annotation_schema["$defs"]["token"]["properties"]["label"]["enum"]

    assert annotation_labels == list(label_schema.label_to_id)


def test_annotation_schema_rejects_unknown_labels() -> None:
    annotation_schema = _load_json(ANNOTATION_SCHEMA_PATH)
    annotation = deepcopy(_load_json(ANNOTATION_FIXTURE_PATH))
    annotation["pages"][0]["tokens"][0]["label"] = "B-UNKNOWN"

    with pytest.raises(ValidationError):
        Draft202012Validator(annotation_schema).validate(annotation)
