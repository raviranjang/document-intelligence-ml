"""Tests for per-class document-classification metrics."""

import pytest

from document_intelligence.classification import DocumentLabel
from document_intelligence.evaluation.classification_metrics import (
    ClassificationSample,
    evaluate_classification,
)


def test_metrics_report_each_class_and_complete_confusion_matrix() -> None:
    samples = (
        ClassificationSample("invoice-tp", DocumentLabel.INVOICE, DocumentLabel.INVOICE),
        ClassificationSample("invoice-fn", DocumentLabel.INVOICE, DocumentLabel.NOT_INVOICE),
        ClassificationSample(
            "non-invoice-tn", DocumentLabel.NOT_INVOICE, DocumentLabel.NOT_INVOICE
        ),
        ClassificationSample("non-invoice-fp", DocumentLabel.NOT_INVOICE, DocumentLabel.INVOICE),
    )

    metrics = evaluate_classification(samples)

    assert metrics.sample_count == 4
    assert metrics.correct == 2
    assert metrics.accuracy == 0.5
    assert metrics.macro_f1 == 0.5
    assert len(metrics.confusion_matrix) == 4
    assert all(class_metrics.precision == 0.5 for class_metrics in metrics.classes)
    assert all(class_metrics.recall == 0.5 for class_metrics in metrics.classes)
    assert all(class_metrics.f1 == 0.5 for class_metrics in metrics.classes)


def test_metrics_preserve_undefined_class_denominators() -> None:
    metrics = evaluate_classification(
        (ClassificationSample("only-invoice", DocumentLabel.INVOICE, DocumentLabel.INVOICE),)
    )

    invoice, not_invoice = metrics.classes
    assert invoice.precision == 1.0
    assert invoice.recall == 1.0
    assert not_invoice.precision is None
    assert not_invoice.recall is None
    assert not_invoice.f1 is None
    assert metrics.macro_f1 == 1.0


def test_metrics_require_samples() -> None:
    with pytest.raises(ValueError, match="requires at least one sample"):
        evaluate_classification(())


def test_sample_rejects_untyped_labels() -> None:
    with pytest.raises(TypeError, match="DocumentLabel"):
        ClassificationSample("sample", "INVOICE", DocumentLabel.INVOICE)  # type: ignore[arg-type]
