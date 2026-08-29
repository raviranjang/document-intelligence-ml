"""Stable, framework-independent document-classification contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any


class DocumentLabel(StrEnum):
    """Supported semantic document types."""

    INVOICE = "INVOICE"
    NOT_INVOICE = "NOT_INVOICE"


@dataclass(frozen=True, slots=True)
class ClassificationModelMetadata:
    """Identity of the classifier that produced a decision."""

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
class DocumentClassification:
    """Versioned classification evidence for one document."""

    document_id: str
    label: DocumentLabel
    decision_score: float
    decision_threshold: float
    matched_signals: tuple[str, ...]
    model: ClassificationModelMetadata
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str):
            raise TypeError("document_id must be a string")
        if not self.document_id.strip():
            raise ValueError("document_id must not be blank")
        if not isinstance(self.label, DocumentLabel):
            raise TypeError("label must be a DocumentLabel")
        score = _validate_unit_interval(self.decision_score, "decision_score")
        threshold = _validate_unit_interval(self.decision_threshold, "decision_threshold")
        if not 0.0 < threshold <= 1.0:
            raise ValueError("decision_threshold must be greater than zero")
        expected_label = DocumentLabel.INVOICE if score >= threshold else DocumentLabel.NOT_INVOICE
        if self.label is not expected_label:
            raise ValueError("label must agree with decision_score and decision_threshold")
        if not isinstance(self.matched_signals, tuple):
            raise TypeError("matched_signals must be a tuple")
        if any(not isinstance(signal, str) or not signal for signal in self.matched_signals):
            raise ValueError("matched_signals must contain non-empty strings")
        if len(set(self.matched_signals)) != len(self.matched_signals):
            raise ValueError("matched_signals must not contain duplicates")
        if not isinstance(self.model, ClassificationModelMetadata):
            raise TypeError("model must be ClassificationModelMetadata")
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported DocumentClassification schema_version")
        object.__setattr__(self, "decision_score", score)
        object.__setattr__(self, "decision_threshold", threshold)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the classification contract to JSON-compatible values."""
        return {
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "label": self.label.value,
            "decision_score": self.decision_score,
            "decision_threshold": self.decision_threshold,
            "matched_signals": list(self.matched_signals),
            "model": {
                "name": self.model.name,
                "version": self.model.version,
                "source": self.model.source,
            },
        }


def _validate_unit_interval(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    canonical_value = float(value)
    if not isfinite(canonical_value) or not 0.0 <= canonical_value <= 1.0:
        raise ValueError(f"{field_name} must be finite and between zero and one")
    return canonical_value
