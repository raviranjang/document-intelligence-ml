"""Validation cases for OCR detection and recognition metrics."""

import pytest

from document_intelligence.common.types import BoundingBox
from document_intelligence.evaluation.ocr_metrics import (
    DetectionSample,
    RecognitionSample,
    edit_distance,
    evaluate_detection,
    evaluate_recognition,
    intersection_over_union,
    match_regions,
)


def _box(x_min: float, y_min: float, x_max: float, y_max: float) -> BoundingBox:
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        (("k", "i", "t", "t", "e", "n"), tuple("sitting"), 3),
        (("invoice", "number"), ("invoice", "no"), 1),
        ((), ("extra",), 1),
        (("same",), ("same",), 0),
    ],
)
def test_edit_distance(
    reference: tuple[str, ...], prediction: tuple[str, ...], expected: int
) -> None:
    assert edit_distance(reference, prediction) == expected


def test_recognition_metrics_are_micro_averaged_and_case_sensitive() -> None:
    samples = (
        RecognitionSample(
            sample_id="identifier-1",
            reference="ABC123",
            prediction="ABC12B",
            cohorts=("identifier",),
            identifier_type="invoice_number",
        ),
        RecognitionSample(
            sample_id="phrase-1",
            reference="Total Amount",
            prediction="total Amount",
            cohorts=("clean_scan",),
        ),
    )

    metrics = evaluate_recognition(samples)

    assert metrics.sample_count == 2
    assert metrics.character_edits == 2
    assert metrics.reference_characters == 18
    assert metrics.character_error_rate == pytest.approx(2 / 18)
    assert metrics.word_edits == 2
    assert metrics.reference_words == 3
    assert metrics.word_error_rate == pytest.approx(2 / 3)
    assert metrics.exact_matches == 0
    assert metrics.exact_match_rate == 0.0
    assert metrics.identifier_count == 1
    assert metrics.identifier_exact_matches == 0
    assert metrics.identifier_exact_match_rate == 0.0


def test_recognition_metrics_do_not_invent_identifier_rate() -> None:
    metrics = evaluate_recognition(
        (RecognitionSample(sample_id="sample-1", reference="Invoice", prediction="Invoice"),)
    )

    assert metrics.exact_match_rate == 1.0
    assert metrics.identifier_count == 0
    assert metrics.identifier_exact_match_rate is None


@pytest.mark.parametrize("reference", ["", " ", "\n"])
def test_recognition_sample_rejects_blank_reference(reference: str) -> None:
    with pytest.raises(ValueError, match="non-blank"):
        RecognitionSample(sample_id="sample", reference=reference, prediction="")


def test_intersection_over_union_handles_overlap_and_disjoint_boxes() -> None:
    first = _box(0, 0, 10, 10)

    assert intersection_over_union(first, _box(0, 0, 10, 10)) == 1.0
    assert intersection_over_union(first, _box(20, 20, 30, 30)) == 0.0
    assert intersection_over_union(first, _box(5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_detection_metrics_use_one_to_one_iou_matching() -> None:
    metrics = evaluate_detection(
        (
            DetectionSample(
                sample_id="page-1",
                reference_boxes=(_box(0, 0, 10, 10), _box(20, 20, 30, 30)),
                predicted_boxes=(_box(0, 0, 10, 10), _box(40, 40, 50, 50)),
            ),
        )
    )

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_detection_matching_does_not_reuse_reference_regions() -> None:
    reference = _box(0, 0, 10, 10)
    metrics = evaluate_detection(
        (
            DetectionSample(
                sample_id="page-1",
                reference_boxes=(reference,),
                predicted_boxes=(reference, reference),
            ),
        )
    )

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0


def test_detection_matching_maximizes_valid_one_to_one_pairs() -> None:
    metrics = evaluate_detection(
        (
            DetectionSample(
                sample_id="page-1",
                reference_boxes=(_box(0, 0, 10, 10), _box(8, 0, 18, 10)),
                predicted_boxes=(_box(0, 0, 10, 10), _box(0, 0, 8, 10)),
            ),
        ),
        iou_threshold=0.1,
    )

    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0


def test_region_matching_exposes_assignments_and_unmatched_indices() -> None:
    reference_boxes = (_box(0, 0, 10, 10), _box(20, 20, 30, 30))
    predicted_boxes = (_box(0, 0, 10, 10), _box(40, 40, 50, 50))

    matching = match_regions(reference_boxes, predicted_boxes)

    assert len(matching.matches) == 1
    assert matching.matches[0].reference_index == 0
    assert matching.matches[0].predicted_index == 0
    assert matching.matches[0].intersection_over_union == 1.0
    assert matching.unmatched_reference_indices == (1,)
    assert matching.unmatched_predicted_indices == (1,)


def test_detection_metrics_preserve_undefined_empty_denominators() -> None:
    metrics = evaluate_detection(
        (DetectionSample(sample_id="empty-page", reference_boxes=(), predicted_boxes=()),)
    )

    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.f1 is None


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, float("inf"), float("nan"), True])
def test_detection_metrics_reject_invalid_iou_threshold(threshold: float) -> None:
    sample = DetectionSample(sample_id="page", reference_boxes=(), predicted_boxes=())

    with pytest.raises((TypeError, ValueError), match="iou_threshold"):
        evaluate_detection((sample,), iou_threshold=threshold)


def test_metrics_require_samples() -> None:
    with pytest.raises(ValueError, match="recognition evaluation"):
        evaluate_recognition(())
    with pytest.raises(ValueError, match="detection evaluation"):
        evaluate_detection(())
