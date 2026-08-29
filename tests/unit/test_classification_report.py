"""Tests for lineage-aware classifier evaluation reports."""

from document_intelligence.classification import ClassificationModelMetadata, DocumentLabel
from document_intelligence.evaluation.classification_metrics import ClassificationSample
from document_intelligence.evaluation.classification_report import (
    ClassificationEvaluationProvenance,
    build_classification_metrics_report,
)


def test_report_contains_lineage_aggregate_and_sorted_cohorts() -> None:
    provenance = ClassificationEvaluationProvenance(
        dataset_name="approved-classifier-evaluation",
        dataset_version="1.0.0",
        source_commit="c" * 40,
        evaluation_config_version="1.0.0",
        model=ClassificationModelMetadata(
            name="keyword-invoice-baseline",
            version="1.0.0",
            source="deterministic_rules",
        ),
    )
    samples = (
        ClassificationSample(
            "invoice-1",
            DocumentLabel.INVOICE,
            DocumentLabel.INVOICE,
            cohorts=("clean_scan", "invoice"),
        ),
        ClassificationSample(
            "other-1",
            DocumentLabel.NOT_INVOICE,
            DocumentLabel.INVOICE,
            cohorts=("noisy_scan",),
        ),
    )

    report = build_classification_metrics_report(provenance=provenance, samples=samples).to_dict()

    assert report["dataset"] == {
        "name": "approved-classifier-evaluation",
        "version": "1.0.0",
    }
    assert report["source_commit"] == "c" * 40
    assert report["aggregate"]["sample_count"] == 2
    assert len(report["aggregate"]["classes"]) == 2
    assert len(report["aggregate"]["confusion_matrix"]) == 4
    assert list(report["slices"]) == ["clean_scan", "invoice", "noisy_scan"]
    assert report["slices"]["clean_scan"]["accuracy"] == 1.0
