"""Tests for multiclass calibration transforms and reliability metrics."""

import pytest

from document_intelligence.calibration import (
    CalibrationSample,
    evaluate_calibration,
    temperature_scaled_probabilities,
)


def test_softmax_is_stable_for_large_logits() -> None:
    probabilities = temperature_scaled_probabilities((10_000.0, 9_999.0), temperature=1.0)

    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1]


def test_metrics_include_nll_ece_brier_and_fixed_reliability_bins() -> None:
    metrics = evaluate_calibration(
        (
            CalibrationSample("correct", (2.0, 0.0), 0),
            CalibrationSample("wrong", (2.0, 0.0), 1),
        ),
        bin_count=5,
    )

    assert metrics.sample_count == 2
    assert metrics.class_count == 2
    assert metrics.accuracy == 0.5
    assert metrics.mean_confidence > 0.5
    assert metrics.expected_calibration_error == pytest.approx(metrics.mean_confidence - 0.5)
    assert metrics.negative_log_likelihood > 0
    assert metrics.brier_score > 0
    assert len(metrics.reliability_bins) == 5
    assert sum(item.sample_count for item in metrics.reliability_bins) == 2
    assert any(item.mean_confidence is None for item in metrics.reliability_bins)


def test_uniform_binary_prediction_has_expected_brier_score() -> None:
    metrics = evaluate_calibration((CalibrationSample("uniform", (0.0, 0.0), 0),), bin_count=2)

    assert metrics.brier_score == 0.5
    assert metrics.mean_confidence == 0.5


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), True])
def test_softmax_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises((TypeError, ValueError), match="temperature"):
        temperature_scaled_probabilities((1.0, 0.0), temperature=temperature)


def test_evaluation_rejects_inconsistent_class_counts() -> None:
    with pytest.raises(ValueError, match="same class count"):
        evaluate_calibration(
            (
                CalibrationSample("binary", (1.0, 0.0), 0),
                CalibrationSample("ternary", (1.0, 0.0, -1.0), 0),
            )
        )


def test_sample_rejects_non_finite_logits() -> None:
    with pytest.raises(ValueError, match="finite"):
        CalibrationSample("invalid", (float("inf"), 0.0), 0)


def test_softmax_rejects_non_finite_logits_at_public_boundary() -> None:
    with pytest.raises(ValueError, match="finite"):
        temperature_scaled_probabilities((float("nan"), 0.0), temperature=1.0)
