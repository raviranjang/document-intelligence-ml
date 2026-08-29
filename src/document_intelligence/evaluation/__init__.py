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
__all__ = [
    "DetectionMetrics",
    "DetectionSample",
    "RecognitionMetrics",
    "RecognitionSample",
    "edit_distance",
    "evaluate_detection",
    "evaluate_recognition",
    "intersection_over_union",
]
