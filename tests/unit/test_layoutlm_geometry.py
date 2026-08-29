"""Tests for canonical LayoutLM bounding-box normalization."""

import pytest

from document_intelligence.common.types import BoundingBox
from document_intelligence.extraction.layoutlm import normalize_bounding_box


def test_normalization_maps_image_boundaries_to_full_grid() -> None:
    normalized = normalize_bounding_box(
        BoundingBox(0, 0, 200, 100), image_width=200, image_height=100
    )

    assert normalized.as_tuple() == (0, 0, 1000, 1000)


def test_normalization_uses_floor_minimum_and_ceiling_maximum() -> None:
    normalized = normalize_bounding_box(
        BoundingBox(0.1, 0.1, 0.2, 0.2), image_width=100, image_height=100
    )

    assert normalized.as_tuple() == (1, 1, 2, 2)


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 100), (100, 0), (-1, 100), (100, -1), (True, 100)],
)
def test_normalization_rejects_invalid_image_dimensions(width: int, height: int) -> None:
    with pytest.raises((TypeError, ValueError), match=r"image_width|image_height"):
        normalize_bounding_box(BoundingBox(0, 0, 1, 1), image_width=width, image_height=height)


def test_normalization_rejects_out_of_page_box_without_clipping() -> None:
    with pytest.raises(ValueError, match="within image dimensions"):
        normalize_bounding_box(BoundingBox(0, 0, 101, 100), image_width=100, image_height=100)


def test_source_contract_rejects_invalid_box_before_normalization() -> None:
    with pytest.raises(ValueError, match="x_max"):
        BoundingBox(10, 0, 10, 20)
