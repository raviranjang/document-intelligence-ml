"""Deterministic binary document-classification metrics."""

from __future__ import annotations

from dataclasses import dataclass

from document_intelligence.classification.types import DocumentLabel


@dataclass(frozen=True, slots=True)
class ClassificationSample:
    """Reference and predicted label for one document."""

    sample_id: str
    reference: DocumentLabel
    prediction: DocumentLabel
    cohorts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str):
            raise TypeError("sample_id must be a string")
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be blank")
        if not isinstance(self.reference, DocumentLabel):
            raise TypeError("reference must be a DocumentLabel")
        if not isinstance(self.prediction, DocumentLabel):
            raise TypeError("prediction must be a DocumentLabel")
        if not isinstance(self.cohorts, tuple):
            raise TypeError("cohorts must be a tuple")
        if any(not isinstance(cohort, str) or not cohort.strip() for cohort in self.cohorts):
            raise ValueError("cohorts must contain non-blank strings")
        if len(set(self.cohorts)) != len(self.cohorts):
            raise ValueError("cohorts must not contain duplicates")


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """One-vs-rest metrics for a document label."""

    label: DocumentLabel
    support: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True, slots=True)
class ConfusionCount:
    """One cell in the binary confusion matrix."""

    reference: DocumentLabel
    prediction: DocumentLabel
    count: int


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Per-class classification metrics and complete confusion matrix."""

    sample_count: int
    correct: int
    accuracy: float
    macro_f1: float | None
    classes: tuple[ClassMetrics, ...]
    confusion_matrix: tuple[ConfusionCount, ...]


def evaluate_classification(
    samples: tuple[ClassificationSample, ...],
) -> ClassificationMetrics:
    """Compute per-class precision, recall, F1, and confusion counts."""
    if not samples:
        raise ValueError("classification evaluation requires at least one sample")

    labels = tuple(DocumentLabel)
    confusion = {
        (reference, prediction): sum(
            sample.reference is reference and sample.prediction is prediction for sample in samples
        )
        for reference in labels
        for prediction in labels
    }
    class_metrics: list[ClassMetrics] = []
    for label in labels:
        true_positives = confusion[(label, label)]
        false_positives = sum(
            confusion[(reference, label)] for reference in labels if reference is not label
        )
        false_negatives = sum(
            confusion[(label, prediction)] for prediction in labels if prediction is not label
        )
        support = true_positives + false_negatives
        predicted_count = true_positives + false_positives
        precision = _ratio_or_none(true_positives, predicted_count)
        recall = _ratio_or_none(true_positives, support)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        class_metrics.append(
            ClassMetrics(
                label=label,
                support=support,
                predicted_count=predicted_count,
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    defined_f1_values = tuple(metrics.f1 for metrics in class_metrics if metrics.f1 is not None)
    correct = sum(confusion[(label, label)] for label in labels)
    return ClassificationMetrics(
        sample_count=len(samples),
        correct=correct,
        accuracy=correct / len(samples),
        macro_f1=(sum(defined_f1_values) / len(defined_f1_values) if defined_f1_values else None),
        classes=tuple(class_metrics),
        confusion_matrix=tuple(
            ConfusionCount(
                reference=reference,
                prediction=prediction,
                count=confusion[(reference, prediction)],
            )
            for reference in labels
            for prediction in labels
        ),
    )


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
