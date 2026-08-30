"""Lifecycle-scoped orchestration of independently versioned model adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from document_intelligence.classification import DocumentClassification
from document_intelligence.common.types import OCRDocument
from document_intelligence.extraction import ExtractionDocument
from document_intelligence.inference.contracts import (
    ClassificationInferenceAdapter,
    ExtractionInferenceAdapter,
    OCRInferenceAdapter,
)


@dataclass(frozen=True, slots=True)
class DocumentInference:
    """Stable aggregate output retaining each independent model identity."""

    ocr: OCRDocument
    classification: DocumentClassification
    extraction: ExtractionDocument
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported DocumentInference schema_version")
        if not isinstance(self.ocr, OCRDocument):
            raise TypeError("ocr must be an OCRDocument")
        if not isinstance(self.classification, DocumentClassification):
            raise TypeError("classification must be a DocumentClassification")
        if not isinstance(self.extraction, ExtractionDocument):
            raise TypeError("extraction must be an ExtractionDocument")
        document_ids = {
            self.ocr.document_id,
            self.classification.document_id,
            self.extraction.document_id,
        }
        if len(document_ids) != 1:
            raise ValueError("all inference outputs must reference the same document_id")

    @property
    def document_id(self) -> str:
        """Return the canonical document identifier shared by every output."""
        return self.ocr.document_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize without exposing framework-specific values."""
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "ocr": self.ocr.to_dict(),
            "classification": self.classification.to_dict(),
            "extraction": self.extraction.to_dict(),
        }


class DocumentInferencePipeline:
    """Run adapters that are constructed once for the process lifecycle."""

    def __init__(
        self,
        *,
        ocr: OCRInferenceAdapter,
        classifier: ClassificationInferenceAdapter,
        extractor: ExtractionInferenceAdapter,
    ) -> None:
        self._ocr = ocr
        self._classifier = classifier
        self._extractor = extractor

    def predict(self, document: Path, *, document_id: str | None = None) -> DocumentInference:
        """Run OCR once, then reuse its canonical output for downstream models."""
        ocr_output = self._ocr.predict(document, document_id=document_id)
        classification = self._classifier.predict(ocr_output)
        extraction = self._extractor.predict(ocr_output)
        return DocumentInference(
            ocr=ocr_output,
            classification=classification,
            extraction=extraction,
        )
