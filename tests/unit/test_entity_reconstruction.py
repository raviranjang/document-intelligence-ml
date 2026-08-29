"""Tests for deterministic BIO entity reconstruction."""

from pathlib import Path

import pytest

from document_intelligence.common.types import BoundingBox, OCRToken
from document_intelligence.extraction import (
    EntityType,
    load_semantic_label_schema,
    reconstruct_entity_spans,
)

LABEL_SCHEMA_PATH = Path("configs/extraction/semantic_labels_v1.json")


def _token(text: str, index: int) -> OCRToken:
    return OCRToken(
        text=text,
        bounding_box=BoundingBox(index, 0, index + 1, 1),
        page_index=0,
        token_index=index,
    )


def test_reconstruction_preserves_entities_boundaries_and_raw_token_text() -> None:
    entities = reconstruct_entity_spans(
        tokens=(
            _token("Seller:", 0),
            _token("Example", 1),
            _token("Supplies", 2),
            _token("Total", 3),
            _token("$10.00", 4),
        ),
        label_names=("O", "B-SELLER_NAME", "I-SELLER_NAME", "O", "B-TOTAL_AMOUNT"),
        page_index=0,
        label_schema=load_semantic_label_schema(LABEL_SCHEMA_PATH),
    )

    assert [(entity.entity_type, entity.raw_value) for entity in entities] == [
        (EntityType.SELLER_NAME, "Example Supplies"),
        (EntityType.TOTAL_AMOUNT, "$10.00"),
    ]
    assert [item.token_index for item in entities[0].token_references] == [1, 2]


def test_reconstruction_rejects_token_label_length_mismatch() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        reconstruct_entity_spans(
            tokens=(_token("Invoice", 0),),
            label_names=(),
            page_index=0,
            label_schema=load_semantic_label_schema(LABEL_SCHEMA_PATH),
        )


def test_reconstruction_rejects_invalid_bio_sequence() -> None:
    with pytest.raises(ValueError, match="invalid BIO transition"):
        reconstruct_entity_spans(
            tokens=(_token("INV-001", 0),),
            label_names=("I-INVOICE_NUMBER",),
            page_index=0,
            label_schema=load_semantic_label_schema(LABEL_SCHEMA_PATH),
        )
