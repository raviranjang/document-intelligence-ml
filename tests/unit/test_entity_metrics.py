"""Tests for token-, entity-, and field-level extraction metrics."""

import pytest

from document_intelligence.evaluation.entity_metrics import (
    EntityEvaluationSample,
    evaluate_entities,
)
from document_intelligence.extraction import EntitySpan, EntityType, TokenReference


def _span(entity_type: EntityType, raw_value: str, *token_indexes: int) -> EntitySpan:
    return EntitySpan(
        entity_type=entity_type,
        raw_value=raw_value,
        token_references=tuple(TokenReference(0, index) for index in token_indexes),
    )


def test_metrics_keep_token_boundary_and_raw_field_matching_distinct() -> None:
    metrics = evaluate_entities(
        (
            EntityEvaluationSample(
                sample_id="invoice-1",
                reference_entities=(
                    _span(EntityType.INVOICE_NUMBER, "INV 001", 0, 1),
                    _span(EntityType.TOTAL_AMOUNT, "$10.00", 3),
                ),
                predicted_entities=(
                    _span(EntityType.INVOICE_NUMBER, "INV OO1", 0, 1),
                    _span(EntityType.TOTAL_AMOUNT, "$10.00", 3),
                    _span(EntityType.SELLER_NAME, "Example", 5),
                ),
            ),
        )
    )

    assert metrics.token.true_positives == 3
    assert metrics.token.false_positives == 1
    assert metrics.token.false_negatives == 0
    assert metrics.token.precision == 0.75
    assert metrics.token.recall == 1.0
    assert metrics.entity.true_positives == 2
    assert metrics.entity.false_positives == 1
    assert metrics.entity.recall == 1.0
    assert metrics.field.field_count == 3
    assert metrics.field.exact_matches == 1
    assert metrics.field.exact_match_rate == pytest.approx(1 / 3)


def test_metrics_preserve_undefined_denominators_for_empty_documents() -> None:
    metrics = evaluate_entities(
        (EntityEvaluationSample("empty", reference_entities=(), predicted_entities=()),)
    )

    assert metrics.token.precision is None
    assert metrics.token.recall is None
    assert metrics.token.f1 is None
    assert metrics.entity.precision is None
    assert metrics.field.exact_match_rate is None


def test_zero_true_positives_produce_zero_f1_when_denominators_exist() -> None:
    metrics = evaluate_entities(
        (
            EntityEvaluationSample(
                "wrong",
                reference_entities=(_span(EntityType.INVOICE_NUMBER, "INV-1", 0),),
                predicted_entities=(_span(EntityType.TOTAL_AMOUNT, "10", 1),),
            ),
        )
    )

    assert metrics.entity.precision == 0.0
    assert metrics.entity.recall == 0.0
    assert metrics.entity.f1 == 0.0


def test_sample_rejects_overlapping_entity_spans() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        EntityEvaluationSample(
            "overlap",
            reference_entities=(
                _span(EntityType.INVOICE_NUMBER, "INV", 0),
                _span(EntityType.ORDER_REFERENCE, "PO", 0),
            ),
            predicted_entities=(),
        )


def test_metrics_require_samples() -> None:
    with pytest.raises(ValueError, match="requires at least one sample"):
        evaluate_entities(())
