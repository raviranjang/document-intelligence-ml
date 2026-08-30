"""Held-out, lineage-aware calibration evaluation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from document_intelligence.calibration.metrics import (
    CalibrationMetrics,
    CalibrationSample,
    evaluate_calibration,
)
from document_intelligence.calibration.temperature import TemperatureCalibrationArtifact
from document_intelligence.evaluation.versioning import (
    COMMIT_PATTERN,
    SEMANTIC_VERSION_PATTERN,
)


@dataclass(frozen=True, slots=True)
class CalibrationEvaluationProvenance:
    """Lineage of the held-out dataset used to evaluate calibration."""

    evaluation_dataset_name: str
    evaluation_dataset_version: str
    source_commit: str
    evaluation_config_version: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.evaluation_dataset_name, str)
            or not self.evaluation_dataset_name.strip()
        ):
            raise ValueError("evaluation_dataset_name must be a non-blank string")
        for field_name in ("evaluation_dataset_version", "evaluation_config_version"):
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
class CalibrationMetricsReport:
    """Held-out aggregate and cohort calibration measurements."""

    provenance: CalibrationEvaluationProvenance
    artifact: TemperatureCalibrationArtifact
    aggregate: CalibrationMetrics
    slices: dict[str, CalibrationMetrics]
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported CalibrationMetricsReport schema_version")

    def to_dict(self) -> dict[str, Any]:
        """Serialize aggregate metrics without raw logits or sample identifiers."""
        artifact_provenance = self.artifact.provenance
        return {
            "schema_version": self.schema_version,
            "evaluation_dataset": {
                "name": self.provenance.evaluation_dataset_name,
                "version": self.provenance.evaluation_dataset_version,
            },
            "source_commit": self.provenance.source_commit,
            "evaluation_config_version": self.provenance.evaluation_config_version,
            "base_model": {
                "name": artifact_provenance.base_model_name,
                "version": artifact_provenance.base_model_version,
            },
            "calibration": {
                "method": self.artifact.method,
                "temperature": self.artifact.temperature,
                "dataset": {
                    "name": artifact_provenance.calibration_dataset_name,
                    "version": artifact_provenance.calibration_dataset_version,
                },
            },
            "aggregate": asdict(self.aggregate),
            "slices": {cohort: asdict(metrics) for cohort, metrics in sorted(self.slices.items())},
        }


def build_calibration_metrics_report(
    *,
    provenance: CalibrationEvaluationProvenance,
    artifact: TemperatureCalibrationArtifact,
    samples: tuple[CalibrationSample, ...],
) -> CalibrationMetricsReport:
    """Evaluate a fitted artifact on held-out aggregate and cohort samples."""
    bin_count = len(artifact.metrics_after.reliability_bins)
    aggregate = evaluate_calibration(
        samples,
        temperature=artifact.temperature,
        bin_count=bin_count,
    )
    cohort_names = sorted({cohort for sample in samples for cohort in sample.cohorts})
    return CalibrationMetricsReport(
        provenance=provenance,
        artifact=artifact,
        aggregate=aggregate,
        slices={
            cohort: evaluate_calibration(
                tuple(sample for sample in samples if cohort in sample.cohorts),
                temperature=artifact.temperature,
                bin_count=bin_count,
            )
            for cohort in cohort_names
        },
    )
