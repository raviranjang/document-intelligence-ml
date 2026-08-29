"""Lineage-aware semantic entity evaluation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from document_intelligence.evaluation.entity_metrics import (
    EntityEvaluationSample,
    EntityExtractionMetrics,
    evaluate_entities,
)
from document_intelligence.evaluation.versioning import (
    COMMIT_PATTERN,
    SEMANTIC_VERSION_PATTERN,
)
from document_intelligence.extraction.types import ExtractionModelMetadata


@dataclass(frozen=True, slots=True)
class EntityEvaluationProvenance:
    """Immutable lineage required to interpret entity metrics."""

    dataset_name: str
    dataset_version: str
    label_schema_version: str
    source_commit: str
    evaluation_config_version: str
    model: ExtractionModelMetadata

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_version",
            "label_schema_version",
            "evaluation_config_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if SEMANTIC_VERSION_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a semantic version")
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise ValueError("dataset_name must be a non-blank string")
        if not isinstance(self.source_commit, str):
            raise TypeError("source_commit must be a string")
        if COMMIT_PATTERN.fullmatch(self.source_commit) is None:
            raise ValueError("source_commit must be a lowercase Git commit hash")
        if not isinstance(self.model, ExtractionModelMetadata):
            raise TypeError("model must be ExtractionModelMetadata")


@dataclass(frozen=True, slots=True)
class EntityMetricsReport:
    """Aggregate and cohort entity extraction metrics."""

    provenance: EntityEvaluationProvenance
    aggregate: EntityExtractionMetrics
    slices: dict[str, EntityExtractionMetrics]
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported EntityMetricsReport schema_version")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to JSON-compatible values."""
        return {
            "schema_version": self.schema_version,
            "dataset": {
                "name": self.provenance.dataset_name,
                "version": self.provenance.dataset_version,
            },
            "label_schema_version": self.provenance.label_schema_version,
            "model": asdict(self.provenance.model),
            "source_commit": self.provenance.source_commit,
            "evaluation_config_version": self.provenance.evaluation_config_version,
            "aggregate": _serialize_metrics(self.aggregate),
            "slices": {
                cohort: _serialize_metrics(metrics)
                for cohort, metrics in sorted(self.slices.items())
            },
        }


def build_entity_metrics_report(
    *,
    provenance: EntityEvaluationProvenance,
    samples: tuple[EntityEvaluationSample, ...],
) -> EntityMetricsReport:
    """Build aggregate and cohort metrics from measured entity samples."""
    aggregate = evaluate_entities(samples)
    cohort_names = sorted({cohort for sample in samples for cohort in sample.cohorts})
    return EntityMetricsReport(
        provenance=provenance,
        aggregate=aggregate,
        slices={
            cohort: evaluate_entities(
                tuple(sample for sample in samples if cohort in sample.cohorts)
            )
            for cohort in cohort_names
        },
    )


def _serialize_metrics(metrics: EntityExtractionMetrics) -> dict[str, Any]:
    return {
        "sample_count": metrics.sample_count,
        "token": asdict(metrics.token),
        "entity": asdict(metrics.entity),
        "field": asdict(metrics.field),
        "entity_types": [
            {
                "entity_type": item.entity_type.value,
                "token": asdict(item.token),
                "entity": asdict(item.entity),
                "field": asdict(item.field),
            }
            for item in metrics.entity_types
        ],
    }
