"""Auditable keyword baseline for binary invoice classification."""

from __future__ import annotations

from document_intelligence.classification.config import KeywordClassifierConfig
from document_intelligence.classification.normalization import normalize_classifier_text
from document_intelligence.classification.types import (
    ClassificationModelMetadata,
    DocumentClassification,
    DocumentLabel,
)
from document_intelligence.common.types import OCRDocument


class KeywordInvoiceClassifier:
    """Classify canonical OCR using versioned, deterministic lexical signals."""

    def __init__(self, config: KeywordClassifierConfig) -> None:
        self._config = config
        self._signals = config.invoice_signals

    def predict(self, document: OCRDocument) -> DocumentClassification:
        """Return binary classification evidence without treating score as probability."""
        normalized_pages = tuple(
            normalize_classifier_text(" ".join(token.text for token in page.tokens))
            for page in document.pages
        )
        padded_pages = tuple(f" {page_text} " for page_text in normalized_pages)
        matched_signals = tuple(
            signal
            for signal in self._signals
            if any(f" {signal} " in page_text for page_text in padded_pages)
        )
        decision_score = len(matched_signals) / len(self._signals)
        label = (
            DocumentLabel.INVOICE
            if decision_score >= self._config.decision_threshold
            else DocumentLabel.NOT_INVOICE
        )
        return DocumentClassification(
            document_id=document.document_id,
            label=label,
            decision_score=decision_score,
            decision_threshold=self._config.decision_threshold,
            matched_signals=matched_signals,
            model=ClassificationModelMetadata(
                name=self._config.model_name,
                version=self._config.model_version,
                source=self._config.model_source,
            ),
        )
