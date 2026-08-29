"""Geometry contracts shared by training and inference code."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An immutable axis-aligned bounding box in source-image coordinates.

    Coordinates use the conventional half-open representation: ``x_min`` and
    ``y_min`` identify the top-left edge, while ``x_max`` and ``y_max`` identify
    the bottom-right edge. Coordinate normalization for model inputs belongs in
    a shared transform rather than in this source-geometry contract.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        """Validate and canonicalize the coordinates."""
        for field_name in ("x_min", "y_min", "x_max", "y_max"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                message = f"{field_name} must be a real number"
                raise TypeError(message)
            canonical_value = float(value)
            if not isfinite(canonical_value):
                message = f"{field_name} must be finite"
                raise ValueError(message)
            if canonical_value < 0:
                message = f"{field_name} must be non-negative"
                raise ValueError(message)
            object.__setattr__(self, field_name, canonical_value)

        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")

    @property
    def width(self) -> float:
        """Return the box width in source-image units."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Return the box height in source-image units."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Return the box area in squared source-image units."""
        return self.width * self.height
