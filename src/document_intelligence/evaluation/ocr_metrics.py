"""Deterministic OCR detection and recognition metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from document_intelligence.common.types import BoundingBox


@dataclass(frozen=True, slots=True)
class RecognitionSample:
    """Ground truth and prediction for one recognized text region."""

    sample_id: str
    reference: str
    prediction: str
    cohorts: tuple[str, ...] = ()
    identifier_type: str | None = None

    def __post_init__(self) -> None:
        _validate_sample_id(self.sample_id)
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("reference must be a non-blank string")
        if not isinstance(self.prediction, str):
            raise TypeError("prediction must be a string")
        _validate_cohorts(self.cohorts)
        if self.identifier_type is not None:
            if not isinstance(self.identifier_type, str):
                raise TypeError("identifier_type must be a string or None")
            if not self.identifier_type.strip():
                raise ValueError("identifier_type must not be blank")


@dataclass(frozen=True, slots=True)
class DetectionSample:
    """Ground-truth and predicted regions for one document page."""

    sample_id: str
    reference_boxes: tuple[BoundingBox, ...]
    predicted_boxes: tuple[BoundingBox, ...]
    cohorts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_sample_id(self.sample_id)
        if not isinstance(self.reference_boxes, tuple) or not all(
            isinstance(box, BoundingBox) for box in self.reference_boxes
        ):
            raise TypeError("reference_boxes must be a tuple of BoundingBox values")
        if not isinstance(self.predicted_boxes, tuple) or not all(
            isinstance(box, BoundingBox) for box in self.predicted_boxes
        ):
            raise TypeError("predicted_boxes must be a tuple of BoundingBox values")
        _validate_cohorts(self.cohorts)


@dataclass(frozen=True, slots=True)
class RecognitionMetrics:
    """Corpus-level recognition counts and rates."""

    sample_count: int
    character_edits: int
    reference_characters: int
    character_error_rate: float
    word_edits: int
    reference_words: int
    word_error_rate: float
    exact_matches: int
    exact_match_rate: float
    identifier_count: int
    identifier_exact_matches: int
    identifier_exact_match_rate: float | None


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Corpus-level one-to-one region-matching counts and rates."""

    sample_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None
    iou_threshold: float


@dataclass(frozen=True, slots=True)
class RegionMatch:
    """One reference/prediction assignment accepted at an IoU threshold."""

    reference_index: int
    predicted_index: int
    intersection_over_union: float


@dataclass(frozen=True, slots=True)
class RegionMatching:
    """Detailed one-to-one region assignments for diagnostics and metrics."""

    matches: tuple[RegionMatch, ...]
    unmatched_reference_indices: tuple[int, ...]
    unmatched_predicted_indices: tuple[int, ...]
    iou_threshold: float


def evaluate_recognition(samples: tuple[RecognitionSample, ...]) -> RecognitionMetrics:
    """Compute micro-averaged CER, WER, and exact-match metrics."""
    if not samples:
        raise ValueError("recognition evaluation requires at least one sample")

    character_edits = 0
    reference_characters = 0
    word_edits = 0
    reference_words = 0
    exact_matches = 0
    identifier_count = 0
    identifier_exact_matches = 0

    for sample in samples:
        reference_characters_for_sample = tuple(sample.reference)
        prediction_characters = tuple(sample.prediction)
        reference_words_for_sample = tuple(sample.reference.split())
        prediction_words = tuple(sample.prediction.split())

        character_edits += edit_distance(reference_characters_for_sample, prediction_characters)
        reference_characters += len(reference_characters_for_sample)
        word_edits += edit_distance(reference_words_for_sample, prediction_words)
        reference_words += len(reference_words_for_sample)

        is_exact = sample.reference == sample.prediction
        exact_matches += int(is_exact)
        if sample.identifier_type is not None:
            identifier_count += 1
            identifier_exact_matches += int(is_exact)

    return RecognitionMetrics(
        sample_count=len(samples),
        character_edits=character_edits,
        reference_characters=reference_characters,
        character_error_rate=character_edits / reference_characters,
        word_edits=word_edits,
        reference_words=reference_words,
        word_error_rate=word_edits / reference_words,
        exact_matches=exact_matches,
        exact_match_rate=exact_matches / len(samples),
        identifier_count=identifier_count,
        identifier_exact_matches=identifier_exact_matches,
        identifier_exact_match_rate=(
            identifier_exact_matches / identifier_count if identifier_count else None
        ),
    )


def evaluate_detection(
    samples: tuple[DetectionSample, ...], *, iou_threshold: float = 0.5
) -> DetectionMetrics:
    """Compute detection metrics using maximum-cardinality one-to-one IoU matching."""
    if not samples:
        raise ValueError("detection evaluation requires at least one sample")
    canonical_threshold = _validate_iou_threshold(iou_threshold)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for sample in samples:
        matching = match_regions(
            sample.reference_boxes,
            sample.predicted_boxes,
            iou_threshold=canonical_threshold,
        )
        matched_count = len(matching.matches)
        true_positives += matched_count
        false_positives += len(sample.predicted_boxes) - matched_count
        false_negatives += len(sample.reference_boxes) - matched_count

    precision = _ratio_or_none(true_positives, true_positives + false_positives)
    recall = _ratio_or_none(true_positives, true_positives + false_negatives)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return DetectionMetrics(
        sample_count=len(samples),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        iou_threshold=canonical_threshold,
    )


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Return intersection over union for two axis-aligned boxes."""
    intersection_width = max(0.0, min(first.x_max, second.x_max) - max(first.x_min, second.x_min))
    intersection_height = max(0.0, min(first.y_max, second.y_max) - max(first.y_min, second.y_min))
    intersection_area = intersection_width * intersection_height
    union_area = first.area + second.area - intersection_area
    return intersection_area / union_area


def edit_distance(reference: tuple[str, ...], prediction: tuple[str, ...]) -> int:
    """Return Levenshtein distance using memory linear in the shorter sequence."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous_row = list(range(len(prediction) + 1))
    for reference_index, reference_unit in enumerate(reference, start=1):
        current_row = [reference_index]
        for prediction_index, prediction_unit in enumerate(prediction, start=1):
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[prediction_index] + 1,
                    previous_row[prediction_index - 1] + int(reference_unit != prediction_unit),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def match_regions(
    reference_boxes: tuple[BoundingBox, ...],
    predicted_boxes: tuple[BoundingBox, ...],
    *,
    iou_threshold: float = 0.5,
) -> RegionMatching:
    """Return deterministic maximum-cardinality one-to-one IoU assignments."""
    canonical_threshold = _validate_iou_threshold(iou_threshold)
    candidate_references = [
        _candidate_references(reference_boxes, predicted_box, iou_threshold=canonical_threshold)
        for predicted_box in predicted_boxes
    ]
    matched_prediction_by_reference: dict[int, int] = {}

    def find_match(predicted_index: int, visited_references: set[int]) -> bool:
        for reference_index in candidate_references[predicted_index]:
            if reference_index in visited_references:
                continue
            visited_references.add(reference_index)
            previous_prediction = matched_prediction_by_reference.get(reference_index)
            if previous_prediction is None or find_match(previous_prediction, visited_references):
                matched_prediction_by_reference[reference_index] = predicted_index
                return True
        return False

    for predicted_index in range(len(predicted_boxes)):
        find_match(predicted_index, set())

    matches = tuple(
        RegionMatch(
            reference_index=reference_index,
            predicted_index=predicted_index,
            intersection_over_union=intersection_over_union(
                reference_boxes[reference_index], predicted_boxes[predicted_index]
            ),
        )
        for reference_index, predicted_index in sorted(matched_prediction_by_reference.items())
    )
    matched_references = frozenset(match.reference_index for match in matches)
    matched_predictions = frozenset(match.predicted_index for match in matches)
    return RegionMatching(
        matches=matches,
        unmatched_reference_indices=tuple(
            index for index in range(len(reference_boxes)) if index not in matched_references
        ),
        unmatched_predicted_indices=tuple(
            index for index in range(len(predicted_boxes)) if index not in matched_predictions
        ),
        iou_threshold=canonical_threshold,
    )


def _candidate_references(
    reference_boxes: tuple[BoundingBox, ...],
    predicted_box: BoundingBox,
    *,
    iou_threshold: float,
) -> tuple[int, ...]:
    candidates: list[tuple[float, int]] = []
    for reference_index, reference_box in enumerate(reference_boxes):
        iou = intersection_over_union(reference_box, predicted_box)
        if iou >= iou_threshold:
            candidates.append((iou, reference_index))
    return tuple(reference_index for _, reference_index in sorted(candidates, reverse=True))


def _validate_sample_id(sample_id: str) -> None:
    if not isinstance(sample_id, str):
        raise TypeError("sample_id must be a string")
    if not sample_id.strip():
        raise ValueError("sample_id must not be blank")


def _validate_cohorts(cohorts: tuple[str, ...]) -> None:
    if not isinstance(cohorts, tuple):
        raise TypeError("cohorts must be a tuple")
    if any(not isinstance(cohort, str) or not cohort.strip() for cohort in cohorts):
        raise ValueError("cohorts must contain non-blank strings")
    if len(set(cohorts)) != len(cohorts):
        raise ValueError("cohorts must not contain duplicates")


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _validate_iou_threshold(iou_threshold: float) -> float:
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
        raise TypeError("iou_threshold must be a real number")
    canonical_threshold = float(iou_threshold)
    if not isfinite(canonical_threshold) or not 0.0 < canonical_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and in the interval (0, 1]")
    return canonical_threshold
