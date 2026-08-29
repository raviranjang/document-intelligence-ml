"""Smoke test for the versioned document-classifier baseline."""

from pathlib import Path

from document_intelligence.classification import (
    DocumentLabel,
    KeywordInvoiceClassifier,
    load_keyword_classifier_config,
)
from document_intelligence.common.types import (
    BoundingBox,
    OCRDocument,
    OCRModelMetadata,
    OCRPage,
    OCRToken,
)


def test_classifier_loads_config_and_returns_canonical_output() -> None:
    classifier = KeywordInvoiceClassifier(
        load_keyword_classifier_config(
            Path("configs/classification/keyword_invoice_baseline.toml")
        )
    )
    document = OCRDocument(
        document_id="synthetic-invoice",
        pages=(
            OCRPage(
                page_index=0,
                tokens=(
                    OCRToken("Invoice", BoundingBox(0, 0, 10, 10), page_index=0, token_index=0),
                    OCRToken("Subtotal", BoundingBox(0, 20, 10, 30), page_index=0, token_index=1),
                ),
            ),
        ),
        model=OCRModelMetadata(name="fixture", version="1", source="synthetic"),
    )

    result = classifier.predict(document)

    assert result.label is DocumentLabel.INVOICE
    assert result.schema_version == "1.0.0"
