"""Tests for the deterministic invoice-classification baseline."""

from pathlib import Path

import pytest

from document_intelligence.classification import (
    DocumentLabel,
    KeywordClassifierConfig,
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


def _document(*texts: str) -> OCRDocument:
    return OCRDocument(
        document_id="document-1",
        pages=(
            OCRPage(
                page_index=0,
                tokens=tuple(
                    OCRToken(
                        text=text,
                        bounding_box=BoundingBox(index, 0, index + 1, 1),
                        page_index=0,
                        token_index=index,
                    )
                    for index, text in enumerate(texts)
                ),
            ),
        ),
        model=OCRModelMetadata(name="ocr", version="1", source="test_fixture"),
    )


@pytest.fixture
def config() -> KeywordClassifierConfig:
    return KeywordClassifierConfig(
        schema_version="1.0.0",
        model_name="keyword-invoice-baseline",
        model_version="1.0.0",
        model_source="deterministic_rules",
        decision_threshold=0.5,
        invoice_signals=("invoice", "amount due", "bill to", "subtotal"),
    )


def test_classifier_returns_invoice_evidence_for_exact_normalized_signals(
    config: KeywordClassifierConfig,
) -> None:
    result = KeywordInvoiceClassifier(config).predict(
        _document("INVOICE", "Amount", "Due:", "$42.00")
    )

    assert result.label is DocumentLabel.INVOICE
    assert result.decision_score == 0.5
    assert result.matched_signals == ("invoice", "amount due")
    assert result.model.source == "deterministic_rules"
    assert result.to_dict()["label"] == "INVOICE"


def test_classifier_returns_not_invoice_without_enough_signals(
    config: KeywordClassifierConfig,
) -> None:
    result = KeywordInvoiceClassifier(config).predict(
        _document("This order has already been invoiced")
    )

    assert result.label is DocumentLabel.NOT_INVOICE
    assert result.decision_score == 0.0
    assert result.matched_signals == ()


def test_classifier_handles_empty_ocr_page(config: KeywordClassifierConfig) -> None:
    result = KeywordInvoiceClassifier(config).predict(_document())

    assert result.label is DocumentLabel.NOT_INVOICE
    assert result.decision_score == 0.0


def test_classifier_does_not_match_phrase_across_page_boundary(
    config: KeywordClassifierConfig,
) -> None:
    document = OCRDocument(
        document_id="two-pages",
        pages=(
            OCRPage(
                page_index=0,
                tokens=(OCRToken("Amount", BoundingBox(0, 0, 1, 1), page_index=0),),
            ),
            OCRPage(
                page_index=1,
                tokens=(OCRToken("Due", BoundingBox(0, 0, 1, 1), page_index=1),),
            ),
        ),
        model=OCRModelMetadata(name="ocr", version="1", source="test_fixture"),
    )

    result = KeywordInvoiceClassifier(config).predict(document)

    assert "amount due" not in result.matched_signals


def test_repository_config_is_strict_and_loadable() -> None:
    config_path = Path("configs/classification/keyword_invoice_baseline.toml")

    config = load_keyword_classifier_config(config_path)

    assert config.schema_version == "1.0.0"
    assert config.model_source == "deterministic_rules"
    assert config.invoice_signals == (
        "invoice",
        "invoice date",
        "bill to",
        "amount due",
        "subtotal",
        "tax",
    )


def test_config_rejects_case_insensitive_duplicate_signals() -> None:
    with pytest.raises(ValueError, match="unique"):
        KeywordClassifierConfig(
            schema_version="1.0.0",
            model_name="model",
            model_version="1.0.0",
            model_source="deterministic_rules",
            decision_threshold=0.5,
            invoice_signals=("Invoice", "invoice"),
        )


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, float("nan"), True])
def test_config_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises((TypeError, ValueError), match="decision_threshold"):
        KeywordClassifierConfig(
            schema_version="1.0.0",
            model_name="model",
            model_version="1.0.0",
            model_source="deterministic_rules",
            decision_threshold=threshold,
            invoice_signals=("invoice",),
        )
