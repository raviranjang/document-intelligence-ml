"""Contract tests for shared document-intelligence domain types."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from document_intelligence.common.types import (
    BoundingBox,
    OCRDocument,
    OCRModelMetadata,
    OCRPage,
    OCRToken,
)


def test_bounding_box_exposes_derived_dimensions() -> None:
    bounding_box = BoundingBox(x_min=10, y_min=20, x_max=35.5, y_max=50)

    assert bounding_box == BoundingBox(x_min=10.0, y_min=20.0, x_max=35.5, y_max=50.0)
    assert bounding_box.width == 25.5
    assert bounding_box.height == 30.0
    assert bounding_box.area == 765.0


def test_bounding_box_is_immutable() -> None:
    bounding_box = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)

    with pytest.raises(FrozenInstanceError):
        bounding_box.x_min = 1  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["x_min", "y_min", "x_max", "y_max"])
@pytest.mark.parametrize("invalid_value", [float("inf"), float("-inf"), float("nan")])
def test_bounding_box_rejects_non_finite_coordinates(
    field_name: str, invalid_value: float
) -> None:
    coordinates = {"x_min": 0.0, "y_min": 0.0, "x_max": 10.0, "y_max": 10.0}
    coordinates[field_name] = invalid_value

    with pytest.raises(ValueError, match=f"{field_name} must be finite"):
        BoundingBox(**coordinates)


@pytest.mark.parametrize("field_name", ["x_min", "y_min", "x_max", "y_max"])
@pytest.mark.parametrize("invalid_value", [-1, True, "1"])
def test_bounding_box_rejects_invalid_coordinate_values(
    field_name: str, invalid_value: Any
) -> None:
    coordinates: dict[str, Any] = {
        "x_min": 0,
        "y_min": 0,
        "x_max": 10,
        "y_max": 10,
    }
    coordinates[field_name] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        BoundingBox(**coordinates)


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        ({"x_min": 2, "y_min": 0, "x_max": 2, "y_max": 1}, "x_max"),
        ({"x_min": 3, "y_min": 0, "x_max": 2, "y_max": 1}, "x_max"),
        ({"x_min": 0, "y_min": 2, "x_max": 1, "y_max": 2}, "y_max"),
        ({"x_min": 0, "y_min": 3, "x_max": 1, "y_max": 2}, "y_max"),
    ],
)
def test_bounding_box_rejects_empty_or_inverted_boxes(
    coordinates: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BoundingBox(**coordinates)


def test_ocr_token_captures_text_geometry_confidence_and_order() -> None:
    bounding_box = BoundingBox(x_min=10, y_min=20, x_max=30, y_max=40)

    token = OCRToken(
        text="Invoice 42",
        bounding_box=bounding_box,
        confidence=1,
        page_index=2,
        token_index=7,
    )

    assert token.text == "Invoice 42"
    assert token.bounding_box is bounding_box
    assert token.confidence == 1.0
    assert token.page_index == 2
    assert token.token_index == 7


def test_ocr_token_allows_missing_confidence() -> None:
    token = OCRToken(
        text="Unscored",
        bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
    )

    assert token.confidence is None
    assert token.page_index == 0
    assert token.token_index == 0


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_ocr_token_rejects_blank_text(text: str) -> None:
    with pytest.raises(ValueError, match="text must contain"):
        OCRToken(
            text=text,
            bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("inf"), float("-inf"), float("nan")])
def test_ocr_token_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence"):
        OCRToken(
            text="Invoice",
            bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
            confidence=confidence,
        )


@pytest.mark.parametrize("field_name", ["page_index", "token_index"])
@pytest.mark.parametrize("invalid_value", [-1, True, 1.5])
def test_ocr_token_rejects_invalid_indexes(field_name: str, invalid_value: Any) -> None:
    values: dict[str, Any] = {
        "text": "Invoice",
        "bounding_box": BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
        field_name: invalid_value,
    }

    with pytest.raises((TypeError, ValueError), match=field_name):
        OCRToken(**values)


def test_ocr_document_serializes_framework_independent_contract() -> None:
    token = OCRToken(
        text="Invoice",
        bounding_box=BoundingBox(x_min=1, y_min=2, x_max=30, y_max=40),
        confidence=0.95,
    )
    document = OCRDocument(
        document_id="invoice-1",
        pages=(OCRPage(page_index=0, tokens=(token,)),),
        model=OCRModelMetadata(
            name="detector+recognizer",
            version="framework-1/model-1",
            source="official_pretrained",
        ),
    )

    assert document.to_dict() == {
        "schema_version": "1.0.0",
        "document_id": "invoice-1",
        "model": {
            "name": "detector+recognizer",
            "version": "framework-1/model-1",
            "source": "official_pretrained",
        },
        "pages": [
            {
                "page_index": 0,
                "tokens": [
                    {
                        "text": "Invoice",
                        "bounding_box": {
                            "x_min": 1.0,
                            "y_min": 2.0,
                            "x_max": 30.0,
                            "y_max": 40.0,
                        },
                        "confidence": 0.95,
                        "token_index": 0,
                    }
                ],
            }
        ],
    }


def test_ocr_page_requires_contiguous_token_order() -> None:
    token = OCRToken(
        text="Invoice",
        bounding_box=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
        token_index=1,
    )

    with pytest.raises(ValueError, match="contiguous reading-order"):
        OCRPage(page_index=0, tokens=(token,))


def test_ocr_document_requires_contiguous_page_order() -> None:
    with pytest.raises(ValueError, match="contiguous and zero-based"):
        OCRDocument(
            document_id="invoice-1",
            pages=(OCRPage(page_index=1, tokens=()),),
            model=OCRModelMetadata(name="model", version="1", source="official_pretrained"),
        )
