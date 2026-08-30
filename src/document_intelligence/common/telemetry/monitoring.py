"""Typed inference telemetry without a required observability SDK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Protocol


class InferenceStage(StrEnum):
    """Stable stages used for latency and failure monitoring."""

    OCR = "ocr"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    PIPELINE = "pipeline"


class MonitoringStatus(StrEnum):
    """Outcome of one monitored inference stage."""

    SUCCESS = "success"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class MonitoringRecord:
    """Privacy-conscious record suitable for logs, spans, and metrics."""

    stage: InferenceStage
    status: MonitoringStatus
    duration_seconds: float
    document_id: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_source: str | None = None
    output_count: int | None = None
    output_categories: tuple[str, ...] = ()
    mean_confidence: float | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, InferenceStage):
            raise TypeError("stage must be an InferenceStage")
        if not isinstance(self.status, MonitoringStatus):
            raise TypeError("status must be a MonitoringStatus")
        if isinstance(self.duration_seconds, bool) or not isinstance(
            self.duration_seconds, (int, float)
        ):
            raise TypeError("duration_seconds must be a real number")
        duration = float(self.duration_seconds)
        if not isfinite(duration) or duration < 0.0:
            raise ValueError("duration_seconds must be finite and non-negative")
        object.__setattr__(self, "duration_seconds", duration)

        for field_name in ("document_id", "model_name", "model_version", "model_source"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-blank string or None")
        model_fields = (self.model_name, self.model_version, self.model_source)
        if any(value is None for value in model_fields) and any(
            value is not None for value in model_fields
        ):
            raise ValueError("model identity fields must be provided together")
        if self.output_count is not None:
            if isinstance(self.output_count, bool) or not isinstance(self.output_count, int):
                raise TypeError("output_count must be an integer or None")
            if self.output_count < 0:
                raise ValueError("output_count must be non-negative")
        if not isinstance(self.output_categories, tuple) or any(
            not isinstance(category, str) or not category.strip()
            for category in self.output_categories
        ):
            raise ValueError("output_categories must be a tuple of non-blank strings")
        if len(set(self.output_categories)) != len(self.output_categories):
            raise ValueError("output_categories must not contain duplicates")
        if self.mean_confidence is not None:
            if isinstance(self.mean_confidence, bool) or not isinstance(
                self.mean_confidence, (int, float)
            ):
                raise TypeError("mean_confidence must be a real number or None")
            confidence = float(self.mean_confidence)
            if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("mean_confidence must be finite and between zero and one")
            object.__setattr__(self, "mean_confidence", confidence)
        if self.status is MonitoringStatus.SUCCESS and self.error_type is not None:
            raise ValueError("successful records must not include error_type")
        if self.status is MonitoringStatus.ERROR:
            if not isinstance(self.error_type, str) or not self.error_type.strip():
                raise ValueError("error records require a non-blank error_type")
            if (
                self.output_count is not None
                or self.output_categories
                or self.mean_confidence is not None
            ):
                raise ValueError("error records must not include output observations")


class MonitoringSink(Protocol):
    """Backend boundary implemented by OpenTelemetry or structured-log adapters."""

    def record(self, observation: MonitoringRecord) -> None:
        """Publish one validated monitoring record."""


class NoOpMonitoringSink:
    """Default sink for processes that have not configured observability."""

    def record(self, observation: MonitoringRecord) -> None:
        """Accept a record without retaining document or model data."""
