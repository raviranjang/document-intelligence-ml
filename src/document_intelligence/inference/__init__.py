"""Production-compatible model loading and inference contracts."""

from document_intelligence.common.telemetry import (
    InferenceStage,
    MonitoringRecord,
    MonitoringSink,
    MonitoringStatus,
    NoOpMonitoringSink,
)
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
    "InferenceStage",
    "MonitoringRecord",
    "MonitoringSink",
    "MonitoringStatus",
    "NoOpMonitoringSink",
    "OCRInferenceAdapter",
]
