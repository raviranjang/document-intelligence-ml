"""Deterministic OCR error classification and aggregate reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from document_intelligence.evaluation.ocr_metrics import (
    DetectionSample,
    RecognitionSample,
    match_regions,
)
from document_intelligence.evaluation.ocr_report import EvaluationProvenance


class FailureStage(StrEnum):
    """Pipeline stages used to preserve ownership of measured failures."""

    DETECTION = "detection"
    RECOGNITION = "recognition"
    SEMANTIC_EXTRACTION = "semantic_extraction"


class OCRErrorCategory(StrEnum):
    """Error categories that can be derived from OCR reference data."""

    MISSED_REGION = "missed_region"
    SPURIOUS_REGION = "spurious_region"
    CHARACTER_SUBSTITUTION = "character_substitution"
    CHARACTER_INSERTION = "character_insertion"
    CHARACTER_DELETION = "character_deletion"
    MIXED_CHARACTER_EDITS = "mixed_character_edits"
    IDENTIFIER_MISMATCH = "identifier_mismatch"


@dataclass(frozen=True, slots=True)
class EditCounts:
    """Levenshtein edit-operation counts for one aligned string pair."""

    substitutions: int
    insertions: int
    deletions: int

    @property
    def total(self) -> int:
        """Return the total number of character edits."""
        return self.substitutions + self.insertions + self.deletions


@dataclass(frozen=True, slots=True)
class OCRErrorObservation:
    """One measured error category associated with an evaluation sample."""

    sample_id: str
    stage: FailureStage
    category: OCRErrorCategory
    occurrences: int
    cohorts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.occurrences < 1:
            raise ValueError("occurrences must be positive")


@dataclass(frozen=True, slots=True)
class ErrorCount:
    """Aggregate impact of one stage/category pair."""

    stage: FailureStage
    category: OCRErrorCategory
    affected_samples: int
    occurrences: int


@dataclass(frozen=True, slots=True)
class OCRErrorAnalysisReport:
    """Lineage-aware baseline error report derived only from measured samples."""

    provenance: EvaluationProvenance
    recognition_samples_evaluated: int
    detection_samples_evaluated: int
    affected_samples: int
    errors: tuple[ErrorCount, ...]
    slices: dict[str, tuple[ErrorCount, ...]]
    iou_threshold: float
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported OCRErrorAnalysisReport schema_version")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report without exposing source text or document content."""
        return {
            "schema_version": self.schema_version,
            "dataset": {
                "name": self.provenance.dataset_name,
                "version": self.provenance.dataset_version,
            },
            "model": asdict(self.provenance.model),
            "source_commit": self.provenance.source_commit,
            "evaluation_config_version": self.provenance.evaluation_config_version,
            "iou_threshold": self.iou_threshold,
            "samples": {
                "recognition_evaluated": self.recognition_samples_evaluated,
                "detection_evaluated": self.detection_samples_evaluated,
                "affected": self.affected_samples,
            },
            "errors": [_serialize_error(error) for error in self.errors],
            "slices": {
                cohort: [_serialize_error(error) for error in errors]
                for cohort, errors in sorted(self.slices.items())
            },
        }


def character_edit_counts(reference: str, prediction: str) -> EditCounts:
    """Return a deterministic minimum-edit alignment for two strings."""
    rows = len(reference) + 1
    columns = len(prediction) + 1
    distances = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distances[row][0] = row
    for column in range(columns):
        distances[0][column] = column

    for row, reference_character in enumerate(reference, start=1):
        for column, prediction_character in enumerate(prediction, start=1):
            distances[row][column] = min(
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
                distances[row - 1][column - 1] + int(reference_character != prediction_character),
            )

    substitutions = insertions = deletions = 0
    row = len(reference)
    column = len(prediction)
    while row or column:
        if (
            row
            and column
            and reference[row - 1] == prediction[column - 1]
            and distances[row][column] == distances[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
        elif row and column and distances[row][column] == distances[row - 1][column - 1] + 1:
            substitutions += 1
            row -= 1
            column -= 1
        elif row and distances[row][column] == distances[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1

    return EditCounts(
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
    )


def analyze_ocr_errors(
    *,
    recognition_samples: tuple[RecognitionSample, ...] = (),
    detection_samples: tuple[DetectionSample, ...] = (),
    iou_threshold: float = 0.5,
) -> tuple[OCRErrorObservation, ...]:
    """Classify OCR failures; semantic failures are deliberately not inferred."""
    if not recognition_samples and not detection_samples:
        raise ValueError("OCR error analysis requires recognition or detection samples")

    observations: list[OCRErrorObservation] = []
    for recognition_sample in recognition_samples:
        edits = character_edit_counts(recognition_sample.reference, recognition_sample.prediction)
        edit_category = _edit_category(edits)
        if edit_category is not None:
            observations.append(
                OCRErrorObservation(
                    sample_id=recognition_sample.sample_id,
                    stage=FailureStage.RECOGNITION,
                    category=edit_category,
                    occurrences=edits.total,
                    cohorts=recognition_sample.cohorts,
                )
            )
        if (
            recognition_sample.identifier_type is not None
            and recognition_sample.reference != recognition_sample.prediction
        ):
            observations.append(
                OCRErrorObservation(
                    sample_id=recognition_sample.sample_id,
                    stage=FailureStage.RECOGNITION,
                    category=OCRErrorCategory.IDENTIFIER_MISMATCH,
                    occurrences=1,
                    cohorts=recognition_sample.cohorts,
                )
            )

    for detection_sample in detection_samples:
        matching = match_regions(
            detection_sample.reference_boxes,
            detection_sample.predicted_boxes,
            iou_threshold=iou_threshold,
        )
        if matching.unmatched_reference_indices:
            observations.append(
                OCRErrorObservation(
                    sample_id=detection_sample.sample_id,
                    stage=FailureStage.DETECTION,
                    category=OCRErrorCategory.MISSED_REGION,
                    occurrences=len(matching.unmatched_reference_indices),
                    cohorts=detection_sample.cohorts,
                )
            )
        if matching.unmatched_predicted_indices:
            observations.append(
                OCRErrorObservation(
                    sample_id=detection_sample.sample_id,
                    stage=FailureStage.DETECTION,
                    category=OCRErrorCategory.SPURIOUS_REGION,
                    occurrences=len(matching.unmatched_predicted_indices),
                    cohorts=detection_sample.cohorts,
                )
            )
    return tuple(observations)


def build_ocr_error_analysis_report(
    *,
    provenance: EvaluationProvenance,
    recognition_samples: tuple[RecognitionSample, ...] = (),
    detection_samples: tuple[DetectionSample, ...] = (),
    iou_threshold: float = 0.5,
) -> OCRErrorAnalysisReport:
    """Build ranked aggregate and cohort error counts from measured samples."""
    observations = analyze_ocr_errors(
        recognition_samples=recognition_samples,
        detection_samples=detection_samples,
        iou_threshold=iou_threshold,
    )
    cohort_names = sorted(
        {cohort for sample in recognition_samples for cohort in sample.cohorts}
        | {cohort for sample in detection_samples for cohort in sample.cohorts}
    )
    slices = {
        cohort: _aggregate_errors(tuple(item for item in observations if cohort in item.cohorts))
        for cohort in cohort_names
    }
    affected_samples = len({(item.stage, item.sample_id) for item in observations})
    canonical_threshold = match_regions((), (), iou_threshold=iou_threshold).iou_threshold
    return OCRErrorAnalysisReport(
        provenance=provenance,
        recognition_samples_evaluated=len(recognition_samples),
        detection_samples_evaluated=len(detection_samples),
        affected_samples=affected_samples,
        errors=_aggregate_errors(observations),
        slices=slices,
        iou_threshold=canonical_threshold,
    )


def _edit_category(edits: EditCounts) -> OCRErrorCategory | None:
    populated = sum(
        count > 0 for count in (edits.substitutions, edits.insertions, edits.deletions)
    )
    if populated == 0:
        return None
    if populated > 1:
        return OCRErrorCategory.MIXED_CHARACTER_EDITS
    if edits.substitutions:
        return OCRErrorCategory.CHARACTER_SUBSTITUTION
    if edits.insertions:
        return OCRErrorCategory.CHARACTER_INSERTION
    return OCRErrorCategory.CHARACTER_DELETION


def _aggregate_errors(observations: tuple[OCRErrorObservation, ...]) -> tuple[ErrorCount, ...]:
    grouped: dict[tuple[FailureStage, OCRErrorCategory], list[OCRErrorObservation]] = {}
    for observation in observations:
        grouped.setdefault((observation.stage, observation.category), []).append(observation)
    counts = (
        ErrorCount(
            stage=stage,
            category=category,
            affected_samples=len({item.sample_id for item in items}),
            occurrences=sum(item.occurrences for item in items),
        )
        for (stage, category), items in grouped.items()
    )
    return tuple(
        sorted(
            counts,
            key=lambda item: (-item.occurrences, item.stage.value, item.category.value),
        )
    )


def _serialize_error(error: ErrorCount) -> dict[str, Any]:
    return {
        "stage": error.stage.value,
        "category": error.category.value,
        "affected_samples": error.affected_samples,
        "occurrences": error.occurrences,
    }
