"""Deterministic multiclass temperature fitting and artifact lineage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log
from typing import Any

from document_intelligence.calibration.config import TemperatureScalingConfig
from document_intelligence.calibration.metrics import (
    CalibrationMetrics,
    CalibrationSample,
    evaluate_calibration,
    temperature_scaled_probabilities,
)
from document_intelligence.evaluation.versioning import (
    COMMIT_PATTERN,
    SEMANTIC_VERSION_PATTERN,
)


@dataclass(frozen=True, slots=True)
class CalibrationProvenance:
    """Lineage for fitting a calibration artifact."""

    base_model_name: str
    base_model_version: str
    calibration_dataset_name: str
    calibration_dataset_version: str
    source_commit: str
    calibration_config_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "base_model_name",
            "base_model_version",
            "calibration_dataset_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-blank string")
        for field_name in ("calibration_dataset_version", "calibration_config_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if SEMANTIC_VERSION_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a semantic version")
        if not isinstance(self.source_commit, str):
            raise TypeError("source_commit must be a string")
        if COMMIT_PATTERN.fullmatch(self.source_commit) is None:
            raise ValueError("source_commit must be a lowercase Git commit hash")


@dataclass(frozen=True, slots=True)
class TemperatureCalibrationArtifact:
    """Versioned temperature and calibration-split fitting diagnostics."""

    provenance: CalibrationProvenance
    temperature: float
    metrics_before: CalibrationMetrics
    metrics_after: CalibrationMetrics
    method: str = "temperature_scaling"
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported calibration artifact schema_version")
        if self.method != "temperature_scaling":
            raise ValueError("method must be temperature_scaling")
        if self.metrics_before.temperature != 1.0:
            raise ValueError("metrics_before must use identity temperature 1.0")
        if self.metrics_after.temperature != self.temperature:
            raise ValueError("metrics_after must use the fitted temperature")
        if self.metrics_before.sample_count != self.metrics_after.sample_count:
            raise ValueError("before and after metrics must use the same fitting samples")

    def calibrate(self, logits: tuple[float, ...]) -> tuple[float, ...]:
        """Convert raw logits to calibrated probabilities."""
        return temperature_scaled_probabilities(logits, temperature=self.temperature)

    def to_dict(self) -> dict[str, Any]:
        """Serialize artifact lineage and aggregate diagnostics without sample data."""
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "base_model": {
                "name": self.provenance.base_model_name,
                "version": self.provenance.base_model_version,
            },
            "calibration_dataset": {
                "name": self.provenance.calibration_dataset_name,
                "version": self.provenance.calibration_dataset_version,
            },
            "source_commit": self.provenance.source_commit,
            "calibration_config_version": self.provenance.calibration_config_version,
            "parameters": {"temperature": self.temperature},
            "fit_metrics": {
                "before": asdict(self.metrics_before),
                "after": asdict(self.metrics_after),
            },
        }


def fit_temperature_scaling(
    *,
    samples: tuple[CalibrationSample, ...],
    provenance: CalibrationProvenance,
    config: TemperatureScalingConfig,
) -> TemperatureCalibrationArtifact:
    """Fit one scalar temperature by bounded log-space NLL minimization."""
    before = evaluate_calibration(samples, temperature=1.0, bin_count=config.reliability_bin_count)
    lower = log(config.minimum_temperature)
    upper = log(config.maximum_temperature)
    inverse_golden_ratio = (5**0.5 - 1) / 2
    left = upper - inverse_golden_ratio * (upper - lower)
    right = lower + inverse_golden_ratio * (upper - lower)
    left_loss = _mean_nll(samples, temperature=exp(left))
    right_loss = _mean_nll(samples, temperature=exp(right))
    for _ in range(config.optimization_iterations):
        if left_loss <= right_loss:
            upper = right
            right = left
            right_loss = left_loss
            left = upper - inverse_golden_ratio * (upper - lower)
            left_loss = _mean_nll(samples, temperature=exp(left))
        else:
            lower = left
            left = right
            left_loss = right_loss
            right = lower + inverse_golden_ratio * (upper - lower)
            right_loss = _mean_nll(samples, temperature=exp(right))

    candidate_temperatures = (
        1.0,
        config.minimum_temperature,
        config.maximum_temperature,
        exp((lower + upper) / 2),
    )
    temperature = min(
        candidate_temperatures,
        key=lambda candidate: (_mean_nll(samples, temperature=candidate), candidate),
    )
    after = evaluate_calibration(
        samples,
        temperature=temperature,
        bin_count=config.reliability_bin_count,
    )
    return TemperatureCalibrationArtifact(
        provenance=provenance,
        temperature=temperature,
        metrics_before=before,
        metrics_after=after,
    )


def _mean_nll(samples: tuple[CalibrationSample, ...], *, temperature: float) -> float:
    if not samples:
        raise ValueError("temperature fitting requires at least one sample")
    class_count = len(samples[0].logits)
    if any(len(sample.logits) != class_count for sample in samples):
        raise ValueError("all calibration samples must have the same class count")
    total = 0.0
    for sample in samples:
        scaled_logits = tuple(value / temperature for value in sample.logits)
        maximum = max(scaled_logits)
        log_denominator = maximum + log(sum(exp(value - maximum) for value in scaled_logits))
        total += log_denominator - scaled_logits[sample.target_index]
    return total / len(samples)
