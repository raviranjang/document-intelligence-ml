"""Stable, framework-independent OCR contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from document_intelligence.common.types.geometry import BoundingBox


@dataclass(frozen=True, slots=True)
class OCRToken:
    """A recognized text token and its location in a document.

    Page and token indexes are zero-based. ``confidence`` is optional because
    some OCR adapters cannot provide a calibrated score; when present, it must
    be in the inclusive range from zero to one.
    """

    text: str
    bounding_box: BoundingBox
    confidence: float | None = None
    page_index: int = 0
    token_index: int = 0

    def __post_init__(self) -> None:
        """Enforce invariants at the framework boundary."""
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not self.text.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        if not isinstance(self.bounding_box, BoundingBox):
            raise TypeError("bounding_box must be a BoundingBox")

        for field_name in ("page_index", "token_index"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                message = f"{field_name} must be an integer"
                raise TypeError(message)
            if value < 0:
                message = f"{field_name} must be non-negative"
                raise ValueError(message)

        if self.confidence is None:
            return
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be a real number or None")
        canonical_confidence = float(self.confidence)
        if not isfinite(canonical_confidence):
            raise ValueError("confidence must be finite")
        if not 0.0 <= canonical_confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", canonical_confidence)
