"""Shared source-to-LayoutLM bounding-box normalization."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

from document_intelligence.common.types import BoundingBox


@dataclass(frozen=True, slots=True)
class NormalizedBoundingBox:
    """Integer bounding box on the inclusive LayoutLM coordinate grid."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    scale: int = 1000

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, int):
            raise TypeError("scale must be an integer")
        if self.scale < 1:
            raise ValueError("scale must be positive")
        for field_name in ("x_min", "y_min", "x_max", "y_max"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if not 0 <= value <= self.scale:
                raise ValueError(f"{field_name} must be between zero and scale")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the model-input coordinate order."""
        return self.x_min, self.y_min, self.x_max, self.y_max


def normalize_bounding_box(
    box: BoundingBox,
    *,
    image_width: int,
    image_height: int,
    scale: int = 1000,
) -> NormalizedBoundingBox:
    """Normalize one valid source box without clipping invalid annotations."""
    _validate_dimension(image_width, "image_width")
    _validate_dimension(image_height, "image_height")
    _validate_dimension(scale, "scale")
    if box.x_max > image_width or box.y_max > image_height:
        raise ValueError("bounding box must remain within image dimensions")
    return NormalizedBoundingBox(
        x_min=floor(scale * box.x_min / image_width),
        y_min=floor(scale * box.y_min / image_height),
        x_max=ceil(scale * box.x_max / image_width),
        y_max=ceil(scale * box.y_max / image_height),
        scale=scale,
    )


def _validate_dimension(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be positive")
