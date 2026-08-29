"""Stable, framework-independent semantic extraction contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EntityType(StrEnum):
    """Initial semantic fields produced as document evidence."""

    ORDER_REFERENCE = "ORDER_REFERENCE"
    INVOICE_NUMBER = "INVOICE_NUMBER"
    SELLER_NAME = "SELLER_NAME"
    TOTAL_AMOUNT = "TOTAL_AMOUNT"
    INVOICE_DATE = "INVOICE_DATE"


@dataclass(frozen=True, slots=True)
class TokenReference:
    """Reference to one canonical OCR token."""

    page_index: int
    token_index: int

    def __post_init__(self) -> None:
        for field_name in ("page_index", "token_index"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True)
class ExtractionModelMetadata:
    """Identity of the extractor that produced candidates."""

    name: str
    version: str
    source: str

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank")


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    """Raw deterministic field candidate with traceable OCR evidence."""

    entity_type: EntityType
    raw_value: str
    token_references: tuple[TokenReference, ...]
    rule_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.entity_type, EntityType):
            raise TypeError("entity_type must be an EntityType")
        if not isinstance(self.raw_value, str):
            raise TypeError("raw_value must be a string")
        if not self.raw_value.strip():
            raise ValueError("raw_value must not be blank")
        if not isinstance(self.token_references, tuple):
            raise TypeError("token_references must be a tuple")
        if not self.token_references:
            raise ValueError("token_references must not be empty")
        if not all(isinstance(reference, TokenReference) for reference in self.token_references):
            raise TypeError("token_references must contain TokenReference values")
        if len(set(self.token_references)) != len(self.token_references):
            raise ValueError("token_references must not contain duplicates")
        if not isinstance(self.rule_id, str):
            raise TypeError("rule_id must be a string")
        if not self.rule_id.strip():
            raise ValueError("rule_id must not be blank")


@dataclass(frozen=True, slots=True)
class ExtractionDocument:
    """Versioned extraction evidence for one document."""

    document_id: str
    entities: tuple[ExtractedEntity, ...]
    model: ExtractionModelMetadata
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str):
            raise TypeError("document_id must be a string")
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if not isinstance(self.entities, tuple):
            raise TypeError("entities must be a tuple")
        if not all(isinstance(entity, ExtractedEntity) for entity in self.entities):
            raise TypeError("entities must contain ExtractedEntity values")
        if not isinstance(self.model, ExtractionModelMetadata):
            raise TypeError("model must be ExtractionModelMetadata")
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported ExtractionDocument schema_version")

    def to_dict(self) -> dict[str, Any]:
        """Serialize extraction evidence without adding normalized business values."""
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "model": {
                "name": self.model.name,
                "version": self.model.version,
                "source": self.model.source,
            },
            "entities": [
                {
                    "entity_type": entity.entity_type.value,
                    "raw_value": entity.raw_value,
                    "token_references": [
                        {
                            "page_index": reference.page_index,
                            "token_index": reference.token_index,
                        }
                        for reference in entity.token_references
                    ],
                    "rule_id": entity.rule_id,
                }
                for entity in self.entities
            ],
        }
