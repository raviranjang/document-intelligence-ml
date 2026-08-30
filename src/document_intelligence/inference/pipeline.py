"""Lifecycle-scoped orchestration of independently versioned model adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from document_intelligence.classification import DocumentClassification
from document_intelligence.common.telemetry import (
    InferenceStage,
    MonitoringRecord,
    MonitoringSink,
    MonitoringStatus,
    NoOpMonitoringSink,
)
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
        monitoring: MonitoringSink | None = None,
        include_document_id_in_monitoring: bool = False,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not isinstance(include_document_id_in_monitoring, bool):
            raise TypeError("include_document_id_in_monitoring must be a boolean")
        self._ocr = ocr
        self._classifier = classifier
        self._extractor = extractor
        self._monitoring = monitoring if monitoring is not None else NoOpMonitoringSink()
        self._include_document_id_in_monitoring = include_document_id_in_monitoring
        self._clock = clock

    def predict(self, document: Path, *, document_id: str | None = None) -> DocumentInference:
        """Run OCR once, then reuse its canonical output for downstream models."""
        pipeline_started = self._clock()
        monitoring_document_id = self._monitoring_document_id(document_id or document.name)
        try:
            ocr_output = self._predict_ocr(document, document_id=document_id)
            classification = self._predict_classification(ocr_output)
            extraction = self._predict_extraction(ocr_output)
            output = DocumentInference(
                ocr=ocr_output,
                classification=classification,
                extraction=extraction,
            )
        except Exception as error:
            self._record_error(
                stage=InferenceStage.PIPELINE,
                started_at=pipeline_started,
                document_id=monitoring_document_id,
                error=error,
            )
            raise
        self._monitoring.record(
            MonitoringRecord(
                stage=InferenceStage.PIPELINE,
                status=MonitoringStatus.SUCCESS,
                duration_seconds=self._elapsed(pipeline_started),
                document_id=self._monitoring_document_id(output.document_id),
            )
        )
        return output

    def _predict_ocr(self, document: Path, *, document_id: str | None) -> OCRDocument:
        started_at = self._clock()
        try:
            output = self._ocr.predict(document, document_id=document_id)
        except Exception as error:
            self._record_error(
                stage=InferenceStage.OCR,
                started_at=started_at,
                document_id=self._monitoring_document_id(document_id or document.name),
                error=error,
            )
            raise
        confidences = tuple(
            token.confidence
            for page in output.pages
            for token in page.tokens
            if token.confidence is not None
        )
        self._monitoring.record(
            MonitoringRecord(
                stage=InferenceStage.OCR,
                status=MonitoringStatus.SUCCESS,
                duration_seconds=self._elapsed(started_at),
                document_id=self._monitoring_document_id(output.document_id),
                model_name=output.model.name,
                model_version=output.model.version,
                model_source=output.model.source,
                output_count=sum(len(page.tokens) for page in output.pages),
                mean_confidence=(sum(confidences) / len(confidences) if confidences else None),
            )
        )
        return output

    def _predict_classification(self, document: OCRDocument) -> DocumentClassification:
        started_at = self._clock()
        try:
            output = self._classifier.predict(document)
        except Exception as error:
            self._record_error(
                stage=InferenceStage.CLASSIFICATION,
                started_at=started_at,
                document_id=self._monitoring_document_id(document.document_id),
                error=error,
            )
            raise
        self._monitoring.record(
            MonitoringRecord(
                stage=InferenceStage.CLASSIFICATION,
                status=MonitoringStatus.SUCCESS,
                duration_seconds=self._elapsed(started_at),
                document_id=self._monitoring_document_id(output.document_id),
                model_name=output.model.name,
                model_version=output.model.version,
                model_source=output.model.source,
                output_count=1,
                output_categories=(output.label.value,),
            )
        )
        return output

    def _predict_extraction(self, document: OCRDocument) -> ExtractionDocument:
        started_at = self._clock()
        try:
            output = self._extractor.predict(document)
        except Exception as error:
            self._record_error(
                stage=InferenceStage.EXTRACTION,
                started_at=started_at,
                document_id=self._monitoring_document_id(document.document_id),
                error=error,
            )
            raise
        self._monitoring.record(
            MonitoringRecord(
                stage=InferenceStage.EXTRACTION,
                status=MonitoringStatus.SUCCESS,
                duration_seconds=self._elapsed(started_at),
                document_id=self._monitoring_document_id(output.document_id),
                model_name=output.model.name,
                model_version=output.model.version,
                model_source=output.model.source,
                output_count=len(output.entities),
                output_categories=tuple(
                    sorted({entity.entity_type.value for entity in output.entities})
                ),
            )
        )
        return output

    def _record_error(
        self,
        *,
        stage: InferenceStage,
        started_at: float,
        document_id: str | None,
        error: Exception,
    ) -> None:
        self._monitoring.record(
            MonitoringRecord(
                stage=stage,
                status=MonitoringStatus.ERROR,
                duration_seconds=self._elapsed(started_at),
                document_id=document_id,
                error_type=type(error).__name__,
            )
        )

    def _elapsed(self, started_at: float) -> float:
        return max(0.0, self._clock() - started_at)

    def _monitoring_document_id(self, document_id: str) -> str | None:
        return document_id if self._include_document_id_in_monitoring else None
