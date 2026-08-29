"""Tests for strict annotation loading and LayoutLM page feature building."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from document_intelligence.extraction import load_semantic_label_schema
from document_intelligence.extraction.layoutlm import (
    AnnotationValidationError,
    EntityAnnotationLoader,
    LayoutLMDatasetConfig,
)

ANNOTATION_SCHEMA_PATH = Path("datasets/schemas/entity-annotation.schema.json")
ANNOTATION_FIXTURE_PATH = Path("datasets/fixtures/synthetic-entity-annotation.json")
LABEL_SCHEMA_PATH = Path("configs/extraction/semantic_labels_v1.json")


def _load_fixture() -> dict[str, Any]:
    with ANNOTATION_FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        value: dict[str, Any] = json.load(fixture_file)
    return value


def _loader() -> EntityAnnotationLoader:
    return EntityAnnotationLoader.from_schema_file(
        ANNOTATION_SCHEMA_PATH, load_semantic_label_schema(LABEL_SCHEMA_PATH)
    )


def test_loader_aggregates_index_geometry_text_and_bio_failures() -> None:
    annotation = deepcopy(_load_fixture())
    annotation["pages"][0]["page_index"] = 2
    annotation["pages"][0]["tokens"][0]["token_index"] = 4
    annotation["pages"][0]["tokens"][0]["text"] = " "
    annotation["pages"][0]["tokens"][0]["bounding_box"]["x_max"] = 1200
    annotation["pages"][0]["tokens"][0]["label"] = "I-SELLER_NAME"

    with pytest.raises(AnnotationValidationError) as error_info:
        _loader().validate(annotation)

    issue_codes = {issue.code for issue in error_info.value.issues}
    assert issue_codes == {
        "non_contiguous_page_index",
        "non_contiguous_token_index",
        "blank_token_text",
        "box_outside_page",
        "invalid_bio_sequence",
    }


def test_loader_rejects_structurally_invalid_annotation_before_conversion() -> None:
    annotation = deepcopy(_load_fixture())
    annotation["pages"][0]["tokens"] = []

    with pytest.raises(AnnotationValidationError) as error_info:
        _loader().validate(annotation)

    assert {issue.code for issue in error_info.value.issues} == {"schema_violation"}


def test_loader_reports_invalid_box_shape_as_domain_issue() -> None:
    annotation = deepcopy(_load_fixture())
    annotation["pages"][0]["tokens"][0]["bounding_box"]["x_min"] = 100
    annotation["pages"][0]["tokens"][0]["bounding_box"]["x_max"] = 50

    with pytest.raises(AnnotationValidationError) as error_info:
        _loader().validate(annotation)

    assert error_info.value.issues[0].code == "invalid_bounding_box"


def test_dataset_config_rejects_training_inference_skew() -> None:
    with pytest.raises(ValueError, match="subword_label_policy"):
        LayoutLMDatasetConfig(
            schema_version="1.0.0",
            max_length=512,
            bounding_box_scale=1000,
            ignore_label_id=-100,
            subword_label_policy="first_piece_only",
            truncation_side="right",
        )
