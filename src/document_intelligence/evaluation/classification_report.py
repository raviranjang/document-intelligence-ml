"""Lineage-aware document-classification evaluation reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from document_intelligence.classification.types import ClassificationModelMetadata
from document_intelligence.evaluation.classification_metrics import (
    ClassificationMetrics,
    ClassificationSample,
    evaluate_classification,
)
from document_intelligence.evaluation.versioning import (
    COMMIT_PATTERN,
    SEMANTIC_VERSION_PATTERN,
)


@dataclass(frozen=True, slots=True)
class ClassificationEvaluationProvenance:
    """Immutable lineage required to interpret classifier metrics."""

    dataset_name: str
    dataset_version: str
    source_commit: str
    evaluation_config_version: str
    model: ClassificationModelMetadata

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
        if not isinstance(self.model, ClassificationModelMetadata):
            raise TypeError("model must be ClassificationModelMetadata")


@dataclass(frozen=True, slots=True)
class ClassificationMetricsReport:
    """Aggregate and cohort metrics for the binary classifier."""

    provenance: ClassificationEvaluationProvenance
    aggregate: ClassificationMetrics
    slices: dict[str, ClassificationMetrics]
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported ClassificationMetricsReport schema_version")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to JSON-compatible values."""
        return {
            "schema_version": self.schema_version,
            "dataset": {
                "name": self.provenance.dataset_name,
                "version": self.provenance.dataset_version,
            },
            "model": {
                "name": self.provenance.model.name,
                "version": self.provenance.model.version,
                "source": self.provenance.model.source,
            },
            "source_commit": self.provenance.source_commit,
            "evaluation_config_version": self.provenance.evaluation_config_version,
            "aggregate": _serialize_metrics(self.aggregate),
            "slices": {
                cohort: _serialize_metrics(metrics)
                for cohort, metrics in sorted(self.slices.items())
            },
        }


def build_classification_metrics_report(
    *,
    provenance: ClassificationEvaluationProvenance,
    samples: tuple[ClassificationSample, ...],
) -> ClassificationMetricsReport:
    """Build aggregate and cohort classifier metrics from measured samples."""
    aggregate = evaluate_classification(samples)
    cohort_names = sorted({cohort for sample in samples for cohort in sample.cohorts})
    return ClassificationMetricsReport(
        provenance=provenance,
        aggregate=aggregate,
        slices={
            cohort: evaluate_classification(
                tuple(sample for sample in samples if cohort in sample.cohorts)
            )
            for cohort in cohort_names
        },
    )


def _serialize_metrics(metrics: ClassificationMetrics) -> dict[str, Any]:
    return {
        "sample_count": metrics.sample_count,
        "correct": metrics.correct,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "classes": [
            {
                "label": class_metrics.label.value,
                "support": class_metrics.support,
                "predicted_count": class_metrics.predicted_count,
                "true_positives": class_metrics.true_positives,
                "false_positives": class_metrics.false_positives,
                "false_negatives": class_metrics.false_negatives,
                "precision": class_metrics.precision,
                "recall": class_metrics.recall,
                "f1": class_metrics.f1,
            }
            for class_metrics in metrics.classes
        ],
        "confusion_matrix": [
            {
                "reference": cell.reference.value,
                "prediction": cell.prediction.value,
                "count": cell.count,
            }
            for cell in metrics.confusion_matrix
        ],
    }
