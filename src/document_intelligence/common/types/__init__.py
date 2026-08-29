"""Stable domain contracts shared across model modules."""

from document_intelligence.common.types.geometry import BoundingBox
from document_intelligence.common.types.ocr import (
    OCRDocument,
    OCRModelMetadata,
    OCRPage,
    OCRToken,
)

__all__ = ["BoundingBox", "OCRDocument", "OCRModelMetadata", "OCRPage", "OCRToken"]
