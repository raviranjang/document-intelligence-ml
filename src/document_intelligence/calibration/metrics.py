"""Multiclass calibration samples, transforms, and reliability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """Raw model logits and reference class for one prediction."""

    sample_id: str
    logits: tuple[float, ...]
    target_index: int
    cohorts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-blank string")
        if not isinstance(self.logits, tuple) or len(self.logits) < 2:
            raise ValueError("logits must contain at least two classes")
        canonical_logits: list[float] = []
        for value in self.logits:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("logits must contain real numbers")
            canonical_value = float(value)
            if not isfinite(canonical_value):
                raise ValueError("logits must be finite")
            canonical_logits.append(canonical_value)
        if isinstance(self.target_index, bool) or not isinstance(self.target_index, int):
            raise TypeError("target_index must be an integer")
        if not 0 <= self.target_index < len(self.logits):
            raise ValueError("target_index must reference a logits class")
        if not isinstance(self.cohorts, tuple):
            raise TypeError("cohorts must be a tuple")
        if any(not isinstance(cohort, str) or not cohort.strip() for cohort in self.cohorts):
            raise ValueError("cohorts must contain non-blank strings")
        if len(set(self.cohorts)) != len(self.cohorts):
            raise ValueError("cohorts must not contain duplicates")
        object.__setattr__(self, "logits", tuple(canonical_logits))


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """Accuracy and mean confidence within one fixed confidence interval."""

    lower_bound: float
    upper_bound: float
    sample_count: int
    mean_confidence: float | None
    accuracy: float | None


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Calibration quality and confidence distribution for a sample set."""

    sample_count: int
    class_count: int
    temperature: float
    negative_log_likelihood: float
    expected_calibration_error: float
    brier_score: float
    mean_confidence: float
    accuracy: float
    reliability_bins: tuple[ReliabilityBin, ...]


def temperature_scaled_probabilities(
    logits: tuple[float, ...], *, temperature: float
) -> tuple[float, ...]:
    """Apply numerically stable softmax after scalar temperature scaling."""
    if len(logits) < 2:
        raise ValueError("logits must contain at least two classes")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number")
    canonical_temperature = float(temperature)
    if not isfinite(canonical_temperature) or canonical_temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    canonical_logits: list[float] = []
    for value in logits:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("logits must contain real numbers")
        canonical_value = float(value)
        if not isfinite(canonical_value):
            raise ValueError("logits must be finite")
        canonical_logits.append(canonical_value)
    scaled_logits = tuple(value / canonical_temperature for value in canonical_logits)
    maximum = max(scaled_logits)
    exponentials = tuple(exp(value - maximum) for value in scaled_logits)
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def evaluate_calibration(
    samples: tuple[CalibrationSample, ...],
    *,
    temperature: float = 1.0,
    bin_count: int = 15,
) -> CalibrationMetrics:
    """Compute NLL, ECE, Brier score, accuracy, and reliability bins."""
    if not samples:
        raise ValueError("calibration evaluation requires at least one sample")
    if isinstance(bin_count, bool) or not isinstance(bin_count, int):
        raise TypeError("bin_count must be an integer")
    if not 2 <= bin_count <= 100:
        raise ValueError("bin_count must be between 2 and 100")
    class_count = len(samples[0].logits)
    if any(len(sample.logits) != class_count for sample in samples):
        raise ValueError("all calibration samples must have the same class count")

    confidences: list[float] = []
    correctness: list[int] = []
    negative_log_likelihood = 0.0
    brier_score = 0.0
    for sample in samples:
        probabilities = temperature_scaled_probabilities(sample.logits, temperature=temperature)
        predicted_index = max(range(class_count), key=probabilities.__getitem__)
        confidence = probabilities[predicted_index]
        confidences.append(confidence)
        correctness.append(int(predicted_index == sample.target_index))
        negative_log_likelihood -= log(max(probabilities[sample.target_index], 1e-300))
        brier_score += sum(
            (probability - int(index == sample.target_index)) ** 2
            for index, probability in enumerate(probabilities)
        )

    bins: list[ReliabilityBin] = []
    expected_calibration_error = 0.0
    for bin_index in range(bin_count):
        lower_bound = bin_index / bin_count
        upper_bound = (bin_index + 1) / bin_count
        members = tuple(
            index
            for index, confidence in enumerate(confidences)
            if min(int(confidence * bin_count), bin_count - 1) == bin_index
        )
        if members:
            mean_confidence = sum(confidences[index] for index in members) / len(members)
            accuracy = sum(correctness[index] for index in members) / len(members)
            expected_calibration_error += (
                len(members) / len(samples) * abs(accuracy - mean_confidence)
            )
        else:
            mean_confidence = None
            accuracy = None
        bins.append(
            ReliabilityBin(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                sample_count=len(members),
                mean_confidence=mean_confidence,
                accuracy=accuracy,
            )
        )

    return CalibrationMetrics(
        sample_count=len(samples),
        class_count=class_count,
        temperature=float(temperature),
        negative_log_likelihood=negative_log_likelihood / len(samples),
        expected_calibration_error=expected_calibration_error,
        brier_score=brier_score / len(samples),
        mean_confidence=sum(confidences) / len(samples),
        accuracy=sum(correctness) / len(samples),
        reliability_bins=tuple(bins),
    )
