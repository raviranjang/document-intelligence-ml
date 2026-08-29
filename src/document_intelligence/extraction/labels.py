"""Versioned BIO label vocabulary for semantic entity extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from document_intelligence.extraction.types import EntityType


class LabelPrefix(StrEnum):
    """Supported BIO sequence prefixes."""

    OUTSIDE = "O"
    BEGIN = "B"
    INSIDE = "I"


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """One stable model label and its integer identifier."""

    label_id: int
    name: str
    prefix: LabelPrefix
    entity_type: EntityType | None

    def __post_init__(self) -> None:
        if isinstance(self.label_id, bool) or not isinstance(self.label_id, int):
            raise TypeError("label_id must be an integer")
        if self.label_id < 0:
            raise ValueError("label_id must be non-negative")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.prefix, LabelPrefix):
            raise TypeError("prefix must be a LabelPrefix")
        if self.prefix is LabelPrefix.OUTSIDE:
            if self.name != "O" or self.entity_type is not None:
                raise ValueError("outside label must be named O and have no entity_type")
        else:
            if not isinstance(self.entity_type, EntityType):
                raise TypeError("entity labels must have an EntityType")
            if self.name != f"{self.prefix.value}-{self.entity_type.value}":
                raise ValueError("entity label name must match its prefix and entity_type")


@dataclass(frozen=True, slots=True)
class SemanticLabelSchema:
    """Complete, versioned label-to-ID mapping for token classification."""

    schema_version: str
    tagging_scheme: str
    labels: tuple[LabelDefinition, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported semantic label schema_version")
        if self.tagging_scheme != "BIO":
            raise ValueError("tagging_scheme must be BIO")
        if not isinstance(self.labels, tuple):
            raise TypeError("labels must be a tuple")
        if not self.labels:
            raise ValueError("labels must not be empty")
        if not all(isinstance(label, LabelDefinition) for label in self.labels):
            raise TypeError("labels must contain LabelDefinition values")
        if tuple(label.label_id for label in self.labels) != tuple(range(len(self.labels))):
            raise ValueError("label_id values must be contiguous and zero-based")
        names = tuple(label.name for label in self.labels)
        if len(set(names)) != len(names):
            raise ValueError("label names must be unique")
        outside_labels = tuple(
            label for label in self.labels if label.prefix is LabelPrefix.OUTSIDE
        )
        if len(outside_labels) != 1 or outside_labels[0].label_id != 0:
            raise ValueError("label ID zero must be the only outside label")
        expected_pairs = {
            (prefix, entity_type)
            for entity_type in EntityType
            for prefix in (LabelPrefix.BEGIN, LabelPrefix.INSIDE)
        }
        actual_pairs = {
            (label.prefix, label.entity_type)
            for label in self.labels
            if label.prefix is not LabelPrefix.OUTSIDE
        }
        if actual_pairs != expected_pairs:
            raise ValueError("labels must define B and I labels for every EntityType")

    @property
    def label_to_id(self) -> dict[str, int]:
        """Return a new name-to-ID mapping safe for caller mutation."""
        return {label.name: label.label_id for label in self.labels}

    @property
    def id_to_label(self) -> dict[int, str]:
        """Return a new ID-to-name mapping safe for caller mutation."""
        return {label.label_id: label.name for label in self.labels}

    def require_label(self, name: str) -> LabelDefinition:
        """Return a known label or fail loudly for unsupported annotation data."""
        for label in self.labels:
            if label.name == name:
                return label
        raise ValueError(f"unsupported semantic label: {name!r}")


def load_semantic_label_schema(path: Path) -> SemanticLabelSchema:
    """Load the strict semantic label vocabulary from JSON."""
    with path.open(encoding="utf-8") as schema_file:
        document: Any = json.load(schema_file)
    if not isinstance(document, dict):
        raise TypeError("semantic label schema root must be an object")
    expected_root_fields = {"schema_version", "tagging_scheme", "labels"}
    _require_exact_fields(document, expected_root_fields, "semantic label schema")
    raw_labels = document["labels"]
    if not isinstance(raw_labels, list):
        raise TypeError("labels must be an array")
    return SemanticLabelSchema(
        schema_version=_require_string(document["schema_version"], "schema_version"),
        tagging_scheme=_require_string(document["tagging_scheme"], "tagging_scheme"),
        labels=tuple(
            _load_label(cast(Any, raw_label), index=index)
            for index, raw_label in enumerate(raw_labels)
        ),
    )


def validate_bio_sequence(label_names: tuple[str, ...], *, schema: SemanticLabelSchema) -> None:
    """Reject invalid BIO transitions in one page-level token sequence."""
    previous: LabelDefinition | None = None
    for index, label_name in enumerate(label_names):
        label = schema.require_label(label_name)
        if label.prefix is LabelPrefix.INSIDE and (
            previous is None
            or previous.prefix is LabelPrefix.OUTSIDE
            or previous.entity_type is not label.entity_type
        ):
            raise ValueError(
                f"invalid BIO transition at token {index}: {label.name!r} does not continue "
                "an entity of the same type"
            )
        previous = label


def _load_label(raw_label: Any, *, index: int) -> LabelDefinition:
    if not isinstance(raw_label, dict):
        raise TypeError(f"labels[{index}] must be an object")
    _require_exact_fields(raw_label, {"id", "name", "prefix", "entity_type"}, f"labels[{index}]")
    raw_id = raw_label["id"]
    if isinstance(raw_id, bool) or not isinstance(raw_id, int):
        raise TypeError("id must be an integer")
    raw_prefix = _require_string(raw_label["prefix"], "prefix")
    try:
        prefix = LabelPrefix(raw_prefix)
    except ValueError as error:
        raise ValueError(f"unsupported label prefix: {raw_prefix!r}") from error
    raw_entity_type = raw_label["entity_type"]
    if raw_entity_type is None:
        entity_type = None
    else:
        entity_type_name = _require_string(raw_entity_type, "entity_type")
        try:
            entity_type = EntityType(entity_type_name)
        except ValueError as error:
            raise ValueError(f"unsupported entity_type: {entity_type_name!r}") from error
    return LabelDefinition(
        label_id=raw_id,
        name=_require_string(raw_label["name"], "name"),
        prefix=prefix,
        entity_type=entity_type,
    )


def _require_exact_fields(
    document: dict[str, Any], expected_fields: set[str], context: str
) -> None:
    unexpected_fields = set(document) - expected_fields
    if unexpected_fields:
        raise ValueError(f"unsupported {context} fields: {sorted(unexpected_fields)}")
    missing_fields = expected_fields - set(document)
    if missing_fields:
        raise ValueError(f"missing {context} fields: {sorted(missing_fields)}")


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value
