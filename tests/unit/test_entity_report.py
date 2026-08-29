"""Tests for lineage-aware semantic extraction reports."""

from document_intelligence.evaluation.entity_metrics import EntityEvaluationSample
from document_intelligence.evaluation.entity_report import (
    EntityEvaluationProvenance,
    build_entity_metrics_report,
)
from document_intelligence.extraction import (
    EntitySpan,
    EntityType,
    ExtractionModelMetadata,
    TokenReference,
)


def test_report_contains_lineage_and_sorted_cohort_metrics() -> None:
    provenance = EntityEvaluationProvenance(
        dataset_name="approved-entity-evaluation",
        dataset_version="1.0.0",
        label_schema_version="1.0.0",
        source_commit="d" * 40,
        evaluation_config_version="1.0.0",
        model=ExtractionModelMetadata(
            name="deterministic-invoice-extractor",
            version="1.0.0",
            source="deterministic_rules",
        ),
    )
    entity = EntitySpan(
        EntityType.INVOICE_NUMBER,
        "INV-001",
        (TokenReference(0, 0),),
    )
    samples = (
        EntityEvaluationSample(
            "invoice-1",
            reference_entities=(entity,),
            predicted_entities=(entity,),
            cohorts=("invoice_number", "clean_scan"),
        ),
        EntityEvaluationSample(
            "invoice-2",
            reference_entities=(),
            predicted_entities=(),
            cohorts=("noisy_scan",),
        ),
    )

    report = build_entity_metrics_report(provenance=provenance, samples=samples).to_dict()

    assert report["dataset"] == {"name": "approved-entity-evaluation", "version": "1.0.0"}
    assert report["label_schema_version"] == "1.0.0"
    assert report["aggregate"]["sample_count"] == 2
    assert report["aggregate"]["entity"]["f1"] == 1.0
    assert len(report["aggregate"]["entity_types"]) == len(EntityType)
    assert list(report["slices"]) == ["clean_scan", "invoice_number", "noisy_scan"]
    assert report["slices"]["noisy_scan"]["entity"]["f1"] is None
