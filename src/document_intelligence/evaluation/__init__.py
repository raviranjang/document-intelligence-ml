"""Aggregate, slice, and regression evaluation."""

from document_intelligence.evaluation.ocr_metrics import (
    DetectionMetrics,
    DetectionSample,
    RecognitionMetrics,
    RecognitionSample,
    edit_distance,
    evaluate_detection,
    evaluate_recognition,
    intersection_over_union,
)
from document_intelligence.evaluation.ocr_report import (
    EvaluationProvenance,
    OCRMetricsReport,
    OCRSliceMetrics,
    build_ocr_metrics_report,
)

__all__ = [
    "DetectionMetrics",
    "DetectionSample",
    "EvaluationProvenance",
    "OCRMetricsReport",
    "OCRSliceMetrics",
    "RecognitionMetrics",
    "RecognitionSample",
    "build_ocr_metrics_report",
    "edit_distance",
    "evaluate_detection",
    "evaluate_recognition",
    "intersection_over_union",
]
