"""Tests for framework-independent inference orchestration."""

from pathlib import Path

import pytest

from document_intelligence.classification import (
    ClassificationModelMetadata,
    DocumentClassification,
    DocumentLabel,
)
from document_intelligence.common.types import (
    BoundingBox,
    OCRDocument,
    OCRModelMetadata,
    OCRPage,
    OCRToken,
)
from document_intelligence.extraction import ExtractionDocument, ExtractionModelMetadata
from document_intelligence.inference import (
    DocumentInference,
    DocumentInferencePipeline,
    InferenceStage,
    MonitoringRecord,
    MonitoringStatus,
)


def _ocr_document(document_id: str = "document-1") -> OCRDocument:
    return OCRDocument(
        document_id=document_id,
        pages=(
            OCRPage(
                page_index=0,
                tokens=(
                    OCRToken(
                        text="INVOICE",
                        bounding_box=BoundingBox(0, 0, 10, 10),
                        confidence=0.8,
                        page_index=0,
                        token_index=0,
                    ),
                ),
            ),
        ),
        model=OCRModelMetadata(name="ocr", version="1.0.0", source="fixture"),
    )


def _classification(document_id: str = "document-1") -> DocumentClassification:
    return DocumentClassification(
        document_id=document_id,
        label=DocumentLabel.INVOICE,
        decision_score=1.0,
        decision_threshold=0.5,
        matched_signals=("invoice",),
        model=ClassificationModelMetadata(name="classifier", version="1.0.0", source="fixture"),
    )


def _extraction(document_id: str = "document-1") -> ExtractionDocument:
    return ExtractionDocument(
        document_id=document_id,
        entities=(),
        model=ExtractionModelMetadata(name="extractor", version="1.0.0", source="fixture"),
    )


class _RecordingOCR:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, str | None]] = []

    def predict(self, document: Path, *, document_id: str | None = None) -> OCRDocument:
        self.calls.append((document, document_id))
        return _ocr_document(document_id or document.name)


class _RecordingClassifier:
    def __init__(self) -> None:
        self.inputs: list[OCRDocument] = []

    def predict(self, document: OCRDocument) -> DocumentClassification:
        self.inputs.append(document)
        return _classification(document.document_id)


class _RecordingExtractor:
    def __init__(self) -> None:
        self.inputs: list[OCRDocument] = []

    def predict(self, document: OCRDocument) -> ExtractionDocument:
        self.inputs.append(document)
        return _extraction(document.document_id)


class _RecordingMonitoringSink:
    def __init__(self) -> None:
        self.records: list[MonitoringRecord] = []

    def record(self, observation: MonitoringRecord) -> None:
        self.records.append(observation)


class _StepClock:
    def __init__(self) -> None:
        self.current = -1.0

    def __call__(self) -> float:
        self.current += 1.0
        return self.current


class _FailingClassifier:
    def predict(self, document: OCRDocument) -> DocumentClassification:
        raise RuntimeError("raw document data must not enter telemetry")


def test_pipeline_reuses_one_ocr_output_for_downstream_adapters() -> None:
    ocr = _RecordingOCR()
    classifier = _RecordingClassifier()
    extractor = _RecordingExtractor()
    pipeline = DocumentInferencePipeline(ocr=ocr, classifier=classifier, extractor=extractor)

    output = pipeline.predict(Path("invoice.png"), document_id="invoice-42")

    assert ocr.calls == [(Path("invoice.png"), "invoice-42")]
    assert classifier.inputs == [output.ocr]
    assert extractor.inputs == [output.ocr]
    assert classifier.inputs[0] is extractor.inputs[0]
    assert output.document_id == "invoice-42"
    assert output.to_dict()["classification"]["model"]["version"] == "1.0.0"


def test_pipeline_keeps_initialized_adapters_across_predictions() -> None:
    ocr = _RecordingOCR()
    pipeline = DocumentInferencePipeline(
        ocr=ocr,
        classifier=_RecordingClassifier(),
        extractor=_RecordingExtractor(),
    )

    first = pipeline.predict(Path("first.png"))
    second = pipeline.predict(Path("second.png"))

    assert [output.document_id for output in (first, second)] == ["first.png", "second.png"]
    assert len(ocr.calls) == 2


def test_pipeline_emits_safe_stage_and_pipeline_observations() -> None:
    monitoring = _RecordingMonitoringSink()
    pipeline = DocumentInferencePipeline(
        ocr=_RecordingOCR(),
        classifier=_RecordingClassifier(),
        extractor=_RecordingExtractor(),
        monitoring=monitoring,
        clock=_StepClock(),
    )

    pipeline.predict(Path("invoice.png"), document_id="invoice-42")

    assert [record.stage for record in monitoring.records] == [
        InferenceStage.OCR,
        InferenceStage.CLASSIFICATION,
        InferenceStage.EXTRACTION,
        InferenceStage.PIPELINE,
    ]
    assert all(record.status is MonitoringStatus.SUCCESS for record in monitoring.records)
    assert monitoring.records[0].mean_confidence == pytest.approx(0.8)
    assert monitoring.records[1].output_categories == ("INVOICE",)
    assert monitoring.records[2].output_count == 0
    assert [record.duration_seconds for record in monitoring.records] == [1.0, 1.0, 1.0, 7.0]
    assert all(record.document_id is None for record in monitoring.records)


def test_pipeline_monitoring_document_id_requires_explicit_opt_in() -> None:
    monitoring = _RecordingMonitoringSink()
    pipeline = DocumentInferencePipeline(
        ocr=_RecordingOCR(),
        classifier=_RecordingClassifier(),
        extractor=_RecordingExtractor(),
        monitoring=monitoring,
        include_document_id_in_monitoring=True,
        clock=_StepClock(),
    )

    pipeline.predict(Path("invoice.png"), document_id="permitted-id")

    assert all(record.document_id == "permitted-id" for record in monitoring.records)


def test_pipeline_records_error_type_without_exception_message() -> None:
    monitoring = _RecordingMonitoringSink()
    pipeline = DocumentInferencePipeline(
        ocr=_RecordingOCR(),
        classifier=_FailingClassifier(),
        extractor=_RecordingExtractor(),
        monitoring=monitoring,
        clock=_StepClock(),
    )

    with pytest.raises(RuntimeError, match="raw document data"):
        pipeline.predict(Path("invoice.png"), document_id="invoice-42")

    assert [record.stage for record in monitoring.records] == [
        InferenceStage.OCR,
        InferenceStage.CLASSIFICATION,
        InferenceStage.PIPELINE,
    ]
    assert monitoring.records[1].status is MonitoringStatus.ERROR
    assert monitoring.records[1].error_type == "RuntimeError"
    assert all("raw document data" not in repr(record) for record in monitoring.records)


def test_aggregate_contract_rejects_mixed_document_outputs() -> None:
    with pytest.raises(ValueError, match="same document_id"):
        DocumentInference(
            ocr=_ocr_document("document-1"),
            classification=_classification("document-2"),
            extraction=_extraction("document-1"),
        )
