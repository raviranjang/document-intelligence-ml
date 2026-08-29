"""Smoke test for the versioned deterministic extraction baseline."""

from pathlib import Path

from document_intelligence.common.types import (
    BoundingBox,
    OCRDocument,
    OCRModelMetadata,
    OCRPage,
    OCRToken,
)
from document_intelligence.extraction import (
    DeterministicExtractor,
    EntityType,
    load_deterministic_extraction_config,
)


def test_extractor_loads_config_and_returns_canonical_evidence() -> None:
    extractor = DeterministicExtractor(
        load_deterministic_extraction_config(
            Path("configs/extraction/deterministic_baseline.toml")
        )
    )
    document = OCRDocument(
        document_id="synthetic-invoice",
        pages=(
            OCRPage(
                page_index=0,
                tokens=(
                    OCRToken(
                        "Invoice No: INV-42",
                        BoundingBox(0, 0, 10, 10),
                        page_index=0,
                    ),
                ),
            ),
        ),
        model=OCRModelMetadata(name="fixture", version="1", source="synthetic"),
    )

    result = extractor.predict(document)

    assert result.schema_version == "1.0.0"
    assert result.entities[0].entity_type is EntityType.INVOICE_NUMBER
    assert result.entities[0].raw_value == "INV-42"
