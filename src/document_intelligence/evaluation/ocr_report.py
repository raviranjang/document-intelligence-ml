"""Versioned, lineage-aware OCR evaluation report generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from document_intelligence.common.types import OCRModelMetadata
from document_intelligence.evaluation.ocr_metrics import (
    DetectionMetrics,
    DetectionSample,
    RecognitionMetrics,
    RecognitionSample,
    evaluate_detection,
    evaluate_recognition,
)
from document_intelligence.evaluation.versioning import (
    COMMIT_PATTERN,
    SEMANTIC_VERSION_PATTERN,
)


@dataclass(frozen=True, slots=True)
class EvaluationProvenance:
    """Immutable lineage required to interpret an evaluation result."""

    dataset_name: str
    dataset_version: str
    source_commit: str
    evaluation_config_version: str
    model: OCRModelMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_name, str):
            raise TypeError("dataset_name must be a string")
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must not be blank")
        if not isinstance(self.dataset_version, str):
            raise TypeError("dataset_version must be a string")
        if SEMANTIC_VERSION_PATTERN.fullmatch(self.dataset_version) is None:
            raise ValueError("dataset_version must be a semantic version")
        if not isinstance(self.source_commit, str):
            raise TypeError("source_commit must be a string")
        if COMMIT_PATTERN.fullmatch(self.source_commit) is None:
            raise ValueError("source_commit must be a lowercase Git commit hash")
        if not isinstance(self.evaluation_config_version, str):
            raise TypeError("evaluation_config_version must be a string")
        if SEMANTIC_VERSION_PATTERN.fullmatch(self.evaluation_config_version) is None:
            raise ValueError("evaluation_config_version must be a semantic version")
        if not isinstance(self.model, OCRModelMetadata):
            raise TypeError("model must be OCRModelMetadata")


@dataclass(frozen=True, slots=True)
class OCRSliceMetrics:
    """Detection and recognition metrics for one named cohort."""

    recognition: RecognitionMetrics | None
    detection: DetectionMetrics | None


@dataclass(frozen=True, slots=True)
class OCRMetricsReport:
    """Canonical OCR metrics report with aggregate and cohort results."""

    provenance: EvaluationProvenance
    aggregate: OCRSliceMetrics
    slices: dict[str, OCRSliceMetrics]
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported OCRMetricsReport schema_version")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to JSON-compatible framework-independent values."""
        return {
            "schema_version": self.schema_version,
            "dataset": {
                "name": self.provenance.dataset_name,
                "version": self.provenance.dataset_version,
            },
            "model": asdict(self.provenance.model),
            "source_commit": self.provenance.source_commit,
            "evaluation_config_version": self.provenance.evaluation_config_version,
            "aggregate": _serialize_slice(self.aggregate),
            "slices": {
                cohort: _serialize_slice(metrics)
                for cohort, metrics in sorted(self.slices.items())
            },
        }


def build_ocr_metrics_report(
    *,
    provenance: EvaluationProvenance,
    recognition_samples: tuple[RecognitionSample, ...] = (),
    detection_samples: tuple[DetectionSample, ...] = (),
    iou_threshold: float = 0.5,
) -> OCRMetricsReport:
    """Build aggregate and cohort metrics without inventing unavailable values."""
    if not recognition_samples and not detection_samples:
        raise ValueError("OCR evaluation requires recognition or detection samples")

    aggregate = OCRSliceMetrics(
        recognition=evaluate_recognition(recognition_samples) if recognition_samples else None,
        detection=(
            evaluate_detection(detection_samples, iou_threshold=iou_threshold)
            if detection_samples
            else None
        ),
    )
    recognition_cohorts = {cohort for sample in recognition_samples for cohort in sample.cohorts}
    detection_cohorts = {cohort for sample in detection_samples for cohort in sample.cohorts}
    cohort_names = sorted(recognition_cohorts | detection_cohorts)
    slices: dict[str, OCRSliceMetrics] = {}
    for cohort in cohort_names:
        recognition_slice = tuple(
            sample for sample in recognition_samples if cohort in sample.cohorts
        )
        detection_slice = tuple(sample for sample in detection_samples if cohort in sample.cohorts)
        slices[cohort] = OCRSliceMetrics(
            recognition=(evaluate_recognition(recognition_slice) if recognition_slice else None),
            detection=(
                evaluate_detection(detection_slice, iou_threshold=iou_threshold)
                if detection_slice
                else None
            ),
        )
    return OCRMetricsReport(provenance=provenance, aggregate=aggregate, slices=slices)


def _serialize_slice(metrics: OCRSliceMetrics) -> dict[str, Any]:
    return {
        "recognition": asdict(metrics.recognition) if metrics.recognition is not None else None,
        "detection": asdict(metrics.detection) if metrics.detection is not None else None,
    }
