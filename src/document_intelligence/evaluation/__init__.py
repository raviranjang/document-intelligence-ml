"""Aggregate, slice, and regression evaluation."""

from document_intelligence.evaluation.classification_metrics import (
    ClassificationMetrics,
    ClassificationSample,
    ClassMetrics,
    ConfusionCount,
    evaluate_classification,
)
from document_intelligence.evaluation.classification_report import (
    ClassificationEvaluationProvenance,
    ClassificationMetricsReport,
    build_classification_metrics_report,
)
from document_intelligence.evaluation.ocr_error_analysis import (
    EditCounts,
    ErrorCount,
    FailureStage,
    OCRErrorAnalysisReport,
    OCRErrorCategory,
    OCRErrorObservation,
    analyze_ocr_errors,
    build_ocr_error_analysis_report,
    character_edit_counts,
)
from document_intelligence.evaluation.ocr_metrics import (
    DetectionMetrics,
    DetectionSample,
    RecognitionMetrics,
    RecognitionSample,
    RegionMatch,
    RegionMatching,
    edit_distance,
    evaluate_detection,
    evaluate_recognition,
    intersection_over_union,
    match_regions,
)
from document_intelligence.evaluation.ocr_report import (
    EvaluationProvenance,
    OCRMetricsReport,
    OCRSliceMetrics,
    build_ocr_metrics_report,
)

__all__ = [
    "ClassMetrics",
    "ClassificationEvaluationProvenance",
    "ClassificationMetrics",
    "ClassificationMetricsReport",
    "ClassificationSample",
    "ConfusionCount",
    "DetectionMetrics",
    "DetectionSample",
    "EditCounts",
    "ErrorCount",
    "EvaluationProvenance",
    "FailureStage",
    "OCRErrorAnalysisReport",
    "OCRErrorCategory",
    "OCRErrorObservation",
    "OCRMetricsReport",
    "OCRSliceMetrics",
    "RecognitionMetrics",
    "RecognitionSample",
    "RegionMatch",
    "RegionMatching",
    "analyze_ocr_errors",
    "build_classification_metrics_report",
    "build_ocr_error_analysis_report",
    "build_ocr_metrics_report",
    "character_edit_counts",
    "edit_distance",
    "evaluate_classification",
    "evaluate_detection",
    "evaluate_recognition",
    "intersection_over_union",
    "match_regions",
]
