"""Token-, entity-, and field-level semantic extraction metrics."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

from document_intelligence.extraction.reconstruction import EntitySpan
from document_intelligence.extraction.types import EntityType

SetItem = TypeVar("SetItem", bound=Hashable)


@dataclass(frozen=True, slots=True)
class EntityEvaluationSample:
    """Reference and predicted entities for one document."""

    sample_id: str
    reference_entities: tuple[EntitySpan, ...]
    predicted_entities: tuple[EntitySpan, ...]
    cohorts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-blank string")
        for field_name in ("reference_entities", "predicted_entities"):
            entities = getattr(self, field_name)
            if not isinstance(entities, tuple) or not all(
                isinstance(entity, EntitySpan) for entity in entities
            ):
                raise TypeError(f"{field_name} must be a tuple of EntitySpan values")
            occupied_tokens = [
                reference for entity in entities for reference in entity.token_references
            ]
            if len(set(occupied_tokens)) != len(occupied_tokens):
                raise ValueError(f"{field_name} must not contain overlapping entity spans")
        if not isinstance(self.cohorts, tuple):
            raise TypeError("cohorts must be a tuple")
        if any(not isinstance(cohort, str) or not cohort.strip() for cohort in self.cohorts):
            raise ValueError("cohorts must contain non-blank strings")
        if len(set(self.cohorts)) != len(self.cohorts):
            raise ValueError("cohorts must not contain duplicates")


@dataclass(frozen=True, slots=True)
class PrecisionRecallF1:
    """Counts and rates for one extraction matching level."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True, slots=True)
class FieldExactMatch:
    """Document/field candidate-set exact match counts."""

    field_count: int
    exact_matches: int
    exact_match_rate: float | None


@dataclass(frozen=True, slots=True)
class EntityTypeMetrics:
    """All evaluation levels for one entity type."""

    entity_type: EntityType
    token: PrecisionRecallF1
    entity: PrecisionRecallF1
    field: FieldExactMatch


@dataclass(frozen=True, slots=True)
class EntityExtractionMetrics:
    """Aggregate extraction metrics with per-field detail."""

    sample_count: int
    token: PrecisionRecallF1
    entity: PrecisionRecallF1
    field: FieldExactMatch
    entity_types: tuple[EntityTypeMetrics, ...]


def evaluate_entities(
    samples: tuple[EntityEvaluationSample, ...],
) -> EntityExtractionMetrics:
    """Compute micro token/entity PRF and raw field exact match."""
    if not samples:
        raise ValueError("entity evaluation requires at least one sample")
    return EntityExtractionMetrics(
        sample_count=len(samples),
        token=_token_metrics(samples),
        entity=_entity_metrics(samples),
        field=_field_metrics(samples),
        entity_types=tuple(
            EntityTypeMetrics(
                entity_type=entity_type,
                token=_token_metrics(samples, entity_type=entity_type),
                entity=_entity_metrics(samples, entity_type=entity_type),
                field=_field_metrics(samples, entity_type=entity_type),
            )
            for entity_type in EntityType
        ),
    )


def _token_metrics(
    samples: tuple[EntityEvaluationSample, ...],
    *,
    entity_type: EntityType | None = None,
) -> PrecisionRecallF1:
    reference = {
        (sample.sample_id, entity.entity_type, token.page_index, token.token_index)
        for sample in samples
        for entity in sample.reference_entities
        for token in entity.token_references
        if entity_type is None or entity.entity_type is entity_type
    }
    prediction = {
        (sample.sample_id, entity.entity_type, token.page_index, token.token_index)
        for sample in samples
        for entity in sample.predicted_entities
        for token in entity.token_references
        if entity_type is None or entity.entity_type is entity_type
    }
    return _set_metrics(reference, prediction)


def _entity_metrics(
    samples: tuple[EntityEvaluationSample, ...],
    *,
    entity_type: EntityType | None = None,
) -> PrecisionRecallF1:
    reference = {
        (sample.sample_id, entity.entity_type, entity.token_references)
        for sample in samples
        for entity in sample.reference_entities
        if entity_type is None or entity.entity_type is entity_type
    }
    prediction = {
        (sample.sample_id, entity.entity_type, entity.token_references)
        for sample in samples
        for entity in sample.predicted_entities
        if entity_type is None or entity.entity_type is entity_type
    }
    return _set_metrics(reference, prediction)


def _field_metrics(
    samples: tuple[EntityEvaluationSample, ...],
    *,
    entity_type: EntityType | None = None,
) -> FieldExactMatch:
    field_count = 0
    exact_matches = 0
    entity_types = (entity_type,) if entity_type is not None else tuple(EntityType)
    for sample in samples:
        for current_type in entity_types:
            reference_values = sorted(
                entity.raw_value
                for entity in sample.reference_entities
                if entity.entity_type is current_type
            )
            prediction_values = sorted(
                entity.raw_value
                for entity in sample.predicted_entities
                if entity.entity_type is current_type
            )
            if not reference_values and not prediction_values:
                continue
            field_count += 1
            exact_matches += int(reference_values == prediction_values)
    return FieldExactMatch(
        field_count=field_count,
        exact_matches=exact_matches,
        exact_match_rate=(exact_matches / field_count if field_count else None),
    )


def _set_metrics(reference: set[SetItem], prediction: set[SetItem]) -> PrecisionRecallF1:
    true_positives = len(reference & prediction)
    false_positives = len(prediction - reference)
    false_negatives = len(reference - prediction)
    precision = _ratio_or_none(true_positives, true_positives + false_positives)
    recall = _ratio_or_none(true_positives, true_positives + false_negatives)
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return PrecisionRecallF1(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
