"""Tests for deterministic field-candidate extraction."""

from pathlib import Path

import pytest

from document_intelligence.common.types import (
    BoundingBox,
    OCRDocument,
    OCRModelMetadata,
    OCRPage,
    OCRToken,
)
from document_intelligence.extraction import (
    DeterministicExtractionConfig,
    DeterministicExtractor,
    EntityType,
    ExtractionRule,
    load_deterministic_extraction_config,
)


def _page(page_index: int, *texts: str) -> OCRPage:
    return OCRPage(
        page_index=page_index,
        tokens=tuple(
            OCRToken(
                text=text,
                bounding_box=BoundingBox(index, 0, index + 1, 1),
                page_index=page_index,
                token_index=index,
            )
            for index, text in enumerate(texts)
        ),
    )


def _document(*pages: OCRPage) -> OCRDocument:
    return OCRDocument(
        document_id="synthetic-invoice",
        pages=pages,
        model=OCRModelMetadata(name="ocr-fixture", version="1", source="synthetic"),
    )


@pytest.fixture
def config() -> DeterministicExtractionConfig:
    return load_deterministic_extraction_config(
        Path("configs/extraction/deterministic_baseline.toml")
    )


def test_baseline_extracts_supported_raw_values_with_token_provenance(
    config: DeterministicExtractionConfig,
) -> None:
    document = _document(
        _page(
            0,
            "Invoice",
            "No:",
            "INV-2026/17",
            "Purchase",
            "Order:",
            "PO-8421",
            "Total",
            "Amount",
            "$",
            "1,234.50",
            "USD",
            "Invoice",
            "Date:",
            "2026-08-29",
        )
    )

    result = DeterministicExtractor(config).predict(document)

    entities = {entity.entity_type: entity for entity in result.entities}
    assert entities[EntityType.INVOICE_NUMBER].raw_value == "INV-2026/17"
    assert entities[EntityType.ORDER_REFERENCE].raw_value == "PO-8421"
    assert entities[EntityType.TOTAL_AMOUNT].raw_value == "$ 1,234.50 USD"
    assert entities[EntityType.INVOICE_DATE].raw_value == "2026-08-29"
    assert [
        reference.token_index for reference in entities[EntityType.TOTAL_AMOUNT].token_references
    ] == [8, 9, 10]
    assert all(
        reference.page_index == 0
        for entity in result.entities
        for reference in entity.token_references
    )
    assert EntityType.SELLER_NAME not in entities
    assert result.model.source == "deterministic_rules"


def test_extractor_returns_all_candidates_without_selecting_business_truth(
    config: DeterministicExtractionConfig,
) -> None:
    result = DeterministicExtractor(config).predict(
        _document(_page(0, "Invoice No: INV-001", "Invoice No: INV-002"))
    )

    assert [entity.raw_value for entity in result.entities] == ["INV-001", "INV-002"]


def test_rules_do_not_match_across_page_boundaries(
    config: DeterministicExtractionConfig,
) -> None:
    result = DeterministicExtractor(config).predict(
        _document(_page(0, "Invoice", "No:"), _page(1, "INV-001"))
    )

    assert result.entities == ()


def test_serialization_preserves_raw_evidence_without_normalized_value(
    config: DeterministicExtractionConfig,
) -> None:
    payload = (
        DeterministicExtractor(config)
        .predict(_document(_page(0, "Amount Due: ₹ 10,000.00")))
        .to_dict()
    )

    assert payload["entities"][0]["raw_value"] == "₹ 10,000.00"
    assert "normalized_value" not in payload["entities"][0]
    assert "confidence" not in payload["entities"][0]


@pytest.mark.parametrize("pattern", ["(", r"invoice (?P<other>\w+)"])
def test_rule_rejects_invalid_or_untraceable_patterns(pattern: str) -> None:
    with pytest.raises(ValueError, match=r"invalid pattern|named 'value'"):
        ExtractionRule(
            rule_id="invalid",
            entity_type=EntityType.INVOICE_NUMBER,
            pattern=pattern,
        )


def test_config_rejects_duplicate_rule_ids() -> None:
    rule = ExtractionRule(
        rule_id="duplicate",
        entity_type=EntityType.INVOICE_NUMBER,
        pattern=r"invoice (?P<value>\w+)",
    )

    with pytest.raises(ValueError, match="rule_id values must be unique"):
        DeterministicExtractionConfig(
            schema_version="1.0.0",
            model_name="baseline",
            model_version="1.0.0",
            model_source="deterministic_rules",
            rules=(rule, rule),
        )
