"""Tests for measured OCR error classification and reporting."""

import pytest

from document_intelligence.common.types import BoundingBox, OCRModelMetadata
from document_intelligence.evaluation.ocr_error_analysis import (
    FailureStage,
    OCRErrorCategory,
    analyze_ocr_errors,
    build_ocr_error_analysis_report,
    character_edit_counts,
)
from document_intelligence.evaluation.ocr_metrics import DetectionSample, RecognitionSample
from document_intelligence.evaluation.ocr_report import EvaluationProvenance


@pytest.fixture
def provenance() -> EvaluationProvenance:
    return EvaluationProvenance(
        dataset_name="approved-ocr-evaluation",
        dataset_version="1.0.0",
        source_commit="b" * 40,
        evaluation_config_version="1.0.0",
        model=OCRModelMetadata(
            name="PP-OCRv6",
            version="paddleocr-3.7.0/paddlepaddle-3.3.1/PP-OCRv6",
            source="official_pretrained",
        ),
    )


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        ("ABC", "AXC", (1, 0, 0)),
        ("ABC", "ABXC", (0, 1, 0)),
        ("ABC", "AC", (0, 0, 1)),
        ("ABC", "AXYZC", (1, 2, 0)),
        ("ABC", "ABC", (0, 0, 0)),
    ],
)
def test_character_edit_counts(
    reference: str, prediction: str, expected: tuple[int, int, int]
) -> None:
    result = character_edit_counts(reference, prediction)

    assert (result.substitutions, result.insertions, result.deletions) == expected
    assert result.total == sum(expected)


def test_analysis_keeps_detection_and_recognition_failures_distinct() -> None:
    box = BoundingBox(0, 0, 10, 10)
    observations = analyze_ocr_errors(
        recognition_samples=(
            RecognitionSample(
                sample_id="identifier-1",
                reference="INV-001",
                prediction="INV-OO1",
                cohorts=("identifier",),
                identifier_type="invoice_number",
            ),
        ),
        detection_samples=(
            DetectionSample(
                sample_id="page-1",
                reference_boxes=(box, BoundingBox(20, 20, 30, 30)),
                predicted_boxes=(box, BoundingBox(40, 40, 50, 50)),
                cohorts=("noisy_scan",),
            ),
        ),
    )

    observed = {(item.stage, item.category): item.occurrences for item in observations}
    assert observed == {
        (FailureStage.RECOGNITION, OCRErrorCategory.CHARACTER_SUBSTITUTION): 2,
        (FailureStage.RECOGNITION, OCRErrorCategory.IDENTIFIER_MISMATCH): 1,
        (FailureStage.DETECTION, OCRErrorCategory.MISSED_REGION): 1,
        (FailureStage.DETECTION, OCRErrorCategory.SPURIOUS_REGION): 1,
    }
    assert all(item.stage is not FailureStage.SEMANTIC_EXTRACTION for item in observations)


def test_exact_recognition_and_matched_regions_create_no_errors() -> None:
    box = BoundingBox(0, 0, 10, 10)

    observations = analyze_ocr_errors(
        recognition_samples=(
            RecognitionSample(sample_id="text-1", reference="Invoice", prediction="Invoice"),
        ),
        detection_samples=(
            DetectionSample(sample_id="page-1", reference_boxes=(box,), predicted_boxes=(box,)),
        ),
    )

    assert observations == ()


def test_report_ranks_errors_and_retains_error_free_cohorts(
    provenance: EvaluationProvenance,
) -> None:
    report = build_ocr_error_analysis_report(
        provenance=provenance,
        recognition_samples=(
            RecognitionSample(
                sample_id="text-1",
                reference="ABC",
                prediction="AXC",
                cohorts=("noisy_scan",),
            ),
            RecognitionSample(
                sample_id="text-2",
                reference="Invoice",
                prediction="Invoice",
                cohorts=("clean_scan",),
            ),
            RecognitionSample(
                sample_id="text-3",
                reference="Total",
                prediction="Tota",
                cohorts=("noisy_scan",),
            ),
        ),
    ).to_dict()

    assert report["samples"] == {
        "recognition_evaluated": 3,
        "detection_evaluated": 0,
        "affected": 2,
    }
    assert [item["category"] for item in report["errors"]] == [
        "character_deletion",
        "character_substitution",
    ]
    assert list(report["slices"]) == ["clean_scan", "noisy_scan"]
    assert report["slices"]["clean_scan"] == []
    assert report["dataset"] == {"name": "approved-ocr-evaluation", "version": "1.0.0"}
    assert "observations" not in report


def test_analysis_requires_measured_samples(provenance: EvaluationProvenance) -> None:
    with pytest.raises(ValueError, match="requires recognition or detection"):
        build_ocr_error_analysis_report(provenance=provenance)


@pytest.mark.parametrize("threshold", [0, 1.1, float("nan"), True])
def test_report_rejects_invalid_iou_threshold(
    provenance: EvaluationProvenance, threshold: float
) -> None:
    sample = RecognitionSample(sample_id="text", reference="A", prediction="A")

    with pytest.raises((TypeError, ValueError), match="iou_threshold"):
        build_ocr_error_analysis_report(
            provenance=provenance,
            recognition_samples=(sample,),
            iou_threshold=threshold,
        )
