"""Production-compatible model loading and inference contracts."""

from document_intelligence.inference.contracts import (
    ClassificationInferenceAdapter,
    ExtractionInferenceAdapter,
    OCRInferenceAdapter,
)
from document_intelligence.inference.pipeline import DocumentInference, DocumentInferencePipeline

__all__ = [
    "ClassificationInferenceAdapter",
    "DocumentInference",
    "DocumentInferencePipeline",
    "ExtractionInferenceAdapter",
    "OCRInferenceAdapter",
]
