"""Deterministic BIO entity reconstruction from source OCR tokens."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from document_intelligence.extraction.labels import (
    LabelPrefix,
    SemanticLabelSchema,
    validate_bio_sequence,
)
from document_intelligence.extraction.types import EntityType, TokenReference


class ReconstructionToken(Protocol):
    """Narrow source-token interface required for entity reconstruction."""

    @property
    def text(self) -> str:
        """Return source token text."""

    @property
    def token_index(self) -> int:
        """Return the zero-based page token index."""


@dataclass(frozen=True, slots=True)
class EntitySpan:
    """One typed, contiguous entity span reconstructed from OCR tokens."""

    entity_type: EntityType
    raw_value: str
    token_references: tuple[TokenReference, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.entity_type, EntityType):
            raise TypeError("entity_type must be an EntityType")
        if not isinstance(self.raw_value, str) or not self.raw_value.strip():
            raise ValueError("raw_value must be a non-blank string")
        if not isinstance(self.token_references, tuple) or not self.token_references:
            raise ValueError("token_references must be a non-empty tuple")
        if not all(isinstance(item, TokenReference) for item in self.token_references):
            raise TypeError("token_references must contain TokenReference values")
        page_indexes = {item.page_index for item in self.token_references}
        if len(page_indexes) != 1:
            raise ValueError("an entity span must remain within one page")
        token_indexes = tuple(item.token_index for item in self.token_references)
        if token_indexes != tuple(range(token_indexes[0], token_indexes[-1] + 1)):
            raise ValueError("entity token references must be contiguous and ordered")


def reconstruct_entity_spans(
    *,
    tokens: Sequence[ReconstructionToken],
    label_names: tuple[str, ...],
    page_index: int,
    label_schema: SemanticLabelSchema,
) -> tuple[EntitySpan, ...]:
    """Reconstruct page entities using one space between canonical OCR tokens."""
    if len(tokens) != len(label_names):
        raise ValueError("tokens and label_names lengths must match")
    if isinstance(page_index, bool) or not isinstance(page_index, int):
        raise TypeError("page_index must be an integer")
    if page_index < 0:
        raise ValueError("page_index must be non-negative")
    if any(token.token_index != index for index, token in enumerate(tokens)):
        raise ValueError("token indexes must be contiguous and zero-based")
    validate_bio_sequence(label_names, schema=label_schema)

    entities: list[EntitySpan] = []
    current_type: EntityType | None = None
    current_tokens: list[ReconstructionToken] = []

    def close_current() -> None:
        nonlocal current_type, current_tokens
        if current_type is not None:
            entities.append(
                EntitySpan(
                    entity_type=current_type,
                    raw_value=" ".join(token.text for token in current_tokens),
                    token_references=tuple(
                        TokenReference(page_index=page_index, token_index=token.token_index)
                        for token in current_tokens
                    ),
                )
            )
        current_type = None
        current_tokens = []

    for token, label_name in zip(tokens, label_names, strict=True):
        label = label_schema.require_label(label_name)
        if label.prefix is LabelPrefix.OUTSIDE:
            close_current()
        elif label.prefix is LabelPrefix.BEGIN:
            close_current()
            current_type = label.entity_type
            current_tokens = [token]
        else:
            current_tokens.append(token)
    close_current()
    return tuple(entities)
