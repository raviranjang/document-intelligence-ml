"""Framework-independent contracts implemented by inference adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from document_intelligence.classification import DocumentClassification
from document_intelligence.common.types import OCRDocument
from document_intelligence.extraction import ExtractionDocument


class OCRInferenceAdapter(Protocol):
    """Convert one supported document file into canonical OCR evidence."""

    def predict(self, document: Path, *, document_id: str | None = None) -> OCRDocument:
        """Return framework-independent OCR output."""


class ClassificationInferenceAdapter(Protocol):
    """Classify one canonical OCR document."""

    def predict(self, document: OCRDocument) -> DocumentClassification:
        """Return versioned document-classification evidence."""


class ExtractionInferenceAdapter(Protocol):
    """Extract semantic evidence from one canonical OCR document."""

    def predict(self, document: OCRDocument) -> ExtractionDocument:
        """Return versioned semantic extraction evidence."""
