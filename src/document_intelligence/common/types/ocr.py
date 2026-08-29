"""Stable, framework-independent OCR contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

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


@dataclass(frozen=True, slots=True)
class OCRModelMetadata:
    """Framework-independent identity for the model that produced OCR evidence."""

    name: str
    version: str
    source: str

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank")


@dataclass(frozen=True, slots=True)
class OCRPage:
    """OCR tokens for one zero-based document page in reading order."""

    page_index: int
    tokens: tuple[OCRToken, ...]

    def __post_init__(self) -> None:
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int):
            raise TypeError("page_index must be an integer")
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if not isinstance(self.tokens, tuple):
            raise TypeError("tokens must be a tuple")
        for expected_index, token in enumerate(self.tokens):
            if not isinstance(token, OCRToken):
                raise TypeError("tokens must contain only OCRToken values")
            if token.page_index != self.page_index:
                raise ValueError("token page_index must match its OCRPage")
            if token.token_index != expected_index:
                raise ValueError("token_index values must be contiguous reading-order indexes")


@dataclass(frozen=True, slots=True)
class OCRDocument:
    """Canonical, versioned OCR output for one document."""

    document_id: str
    pages: tuple[OCRPage, ...]
    model: OCRModelMetadata
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str):
            raise TypeError("document_id must be a string")
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported OCRDocument schema_version")
        if not isinstance(self.model, OCRModelMetadata):
            raise TypeError("model must be OCRModelMetadata")
        if not isinstance(self.pages, tuple):
            raise TypeError("pages must be a tuple")
        if not self.pages:
            raise ValueError("OCRDocument must contain at least one page")
        for expected_index, page in enumerate(self.pages):
            if not isinstance(page, OCRPage):
                raise TypeError("pages must contain only OCRPage values")
            if page.page_index != expected_index:
                raise ValueError("page_index values must be contiguous and zero-based")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canonical contract without framework-specific values."""
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "model": {
                "name": self.model.name,
                "version": self.model.version,
                "source": self.model.source,
            },
            "pages": [
                {
                    "page_index": page.page_index,
                    "tokens": [
                        {
                            "text": token.text,
                            "bounding_box": {
                                "x_min": token.bounding_box.x_min,
                                "y_min": token.bounding_box.y_min,
                                "x_max": token.bounding_box.x_max,
                                "y_max": token.bounding_box.y_max,
                            },
                            "confidence": token.confidence,
                            "token_index": token.token_index,
                        }
                        for token in page.tokens
                    ],
                }
                for page in self.pages
            ],
        }
