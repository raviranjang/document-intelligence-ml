"""Tests for lineage-aware OCR evaluation report generation."""

import pytest

from document_intelligence.common.types import BoundingBox, OCRModelMetadata
from document_intelligence.evaluation.ocr_metrics import DetectionSample, RecognitionSample
from document_intelligence.evaluation.ocr_report import (
    EvaluationProvenance,
    build_ocr_metrics_report,
)


@pytest.fixture
def provenance() -> EvaluationProvenance:
    return EvaluationProvenance(
        dataset_name="synthetic-ocr-evaluation",
        dataset_version="1.0.0",
        source_commit="a" * 40,
        evaluation_config_version="1.0.0",
        model=OCRModelMetadata(
            name="PP-OCRv6_medium_det+PP-OCRv6_medium_rec",
            version="paddleocr-3.7.0/paddlepaddle-3.3.1/PP-OCRv6",
            source="official_pretrained",
        ),
    )


def test_report_contains_aggregate_lineage_and_sorted_slices(
    provenance: EvaluationProvenance,
) -> None:
    recognition_samples = (
        RecognitionSample(
            sample_id="recognition-1",
            reference="INV-001",
            prediction="INV-001",
            cohorts=("identifier", "clean_scan"),
            identifier_type="invoice_number",
        ),
        RecognitionSample(
            sample_id="recognition-2",
            reference="Total",
            prediction="Tota1",
            cohorts=("noisy_scan",),
        ),
    )
    detection_samples = (
        DetectionSample(
            sample_id="page-1",
            reference_boxes=(BoundingBox(0, 0, 10, 10),),
            predicted_boxes=(BoundingBox(0, 0, 10, 10),),
            cohorts=("clean_scan",),
        ),
    )

    report = build_ocr_metrics_report(
        provenance=provenance,
        recognition_samples=recognition_samples,
        detection_samples=detection_samples,
        iou_threshold=0.75,
    ).to_dict()

    assert report["schema_version"] == "1.0.0"
    assert report["dataset"] == {"name": "synthetic-ocr-evaluation", "version": "1.0.0"}
    assert report["source_commit"] == "a" * 40
    assert report["aggregate"]["recognition"]["sample_count"] == 2
    assert report["aggregate"]["detection"]["iou_threshold"] == 0.75
    assert list(report["slices"]) == ["clean_scan", "identifier", "noisy_scan"]
    assert report["slices"]["identifier"]["detection"] is None
    assert report["slices"]["clean_scan"]["recognition"]["exact_match_rate"] == 1.0


def test_report_allows_task_specific_evaluation_without_fabricated_metrics(
    provenance: EvaluationProvenance,
) -> None:
    report = build_ocr_metrics_report(
        provenance=provenance,
        recognition_samples=(
            RecognitionSample(sample_id="sample", reference="Invoice", prediction="Invoice"),
        ),
    ).to_dict()

    assert report["aggregate"]["recognition"]["exact_match_rate"] == 1.0
    assert report["aggregate"]["detection"] is None


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("dataset_version", "latest"),
        ("source_commit", "not-a-commit"),
        ("evaluation_config_version", "v1"),
    ],
)
def test_provenance_rejects_unversioned_lineage(field_name: str, invalid_value: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        EvaluationProvenance(
            dataset_name="dataset",
            dataset_version=(invalid_value if field_name == "dataset_version" else "1.0.0"),
            source_commit=(invalid_value if field_name == "source_commit" else "a" * 40),
            evaluation_config_version=(
                invalid_value if field_name == "evaluation_config_version" else "1.0.0"
            ),
            model=OCRModelMetadata(name="model", version="1", source="official_pretrained"),
        )


def test_report_requires_at_least_one_evaluation_task(
    provenance: EvaluationProvenance,
) -> None:
    with pytest.raises(ValueError, match="requires recognition or detection"):
        build_ocr_metrics_report(provenance=provenance)
